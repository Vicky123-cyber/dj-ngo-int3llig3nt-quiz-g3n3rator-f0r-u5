from pathlib import Path
import random
import uuid

import pymupdf

from django.core.files.base import ContentFile
from django.db import transaction
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone

from .models import (
    UploadedDocument,
    GeneratedQuestion,
    QuizAttempt,
    StudentAnswer,
)
from .nlp_processor import process_text
from .keyword_extractor import extract_keywords
from .question_generator import (
    answers_match,
    generate_fill_blank_questions,
    generate_multiple_choice_questions,
    generate_short_answer_questions,
    generate_true_false_questions,
    short_answer_contains_keyword,
)


MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

COMBINED_PER_TYPE = 3  # questions sampled from each type for combined quiz

SESSION_KEYS = {
    "MCQ": "active_quiz_mcq",
    "FILL_BLANK": "active_quiz_fitb",
    "TRUE_FALSE": "active_quiz_tf",
    "SHORT_ANSWER": "active_quiz_short",
    "COMBINED": "active_quiz_combined",
}

QUIZ_LABELS = {
    "MCQ": "Multiple Choice",
    "FILL_BLANK": "Fill in the Blank",
    "TRUE_FALSE": "True or False",
    "SHORT_ANSWER": "Short Answer",
    "COMBINED": "Combined",
}


def _clear_quiz_session(request):
    """Remove every active quiz key from the session."""

    for key in SESSION_KEYS.values():
        request.session.pop(key, None)

    request.session.pop("active_quiz_document_id", None)


def _store_quiz_session(request, document_id, quiz_by_type):
    """
    Save each question type's quiz list into the session under its
    own key, plus the shared document id.
    """

    for quiz_type, key in SESSION_KEYS.items():
        items = quiz_by_type.get(quiz_type, [])
        request.session[key] = items

    request.session["active_quiz_document_id"] = document_id


def _records_to_session_items(records):
    """Convert GeneratedQuestion rows to session-friendly dicts."""

    items = []
    for record in records:
        items.append({
            "question_id": record.id,
            "question": record.question_text,
            "question_type": record.question_type,
            "options": record.options,
            "answer": record.correct_answer,
            "source_sentence": record.source_sentence,
        })
    return items


def _public_items(items):
    """Strip answers before sending quiz items to the template."""

    return [
        {
            "question_id": item["question_id"],
            "question": item["question"],
            "question_type": item["question_type"],
            "options": item["options"],
        }
        for item in items
    ]


def _build_combined_quiz(saved_by_type):
    """
    Build a mixed quiz by sampling COMBINED_PER_TYPE questions from
    each question type, then shuffling the combined list.
    """

    combined = []

    for quiz_type in ("MCQ", "FILL_BLANK", "TRUE_FALSE", "SHORT_ANSWER"):
        items = saved_by_type.get(quiz_type, []) or []
        if not items:
            continue
        sample_size = min(COMBINED_PER_TYPE, len(items))
        combined.extend(random.sample(items, sample_size))

    random.shuffle(combined)
    return combined


def _build_document_context(request, document_record, skip_ml=False):
    """
    Rebuild the full template context for an already-saved document.

    Re-runs NLP and keyword extraction, loads the document's saved
    questions from the database grouped by type, builds the combined
    quiz, and rebuilds all active quizzes in the session.
    """

    extracted_text = document_record.extracted_text

    nlp_results = process_text(extracted_text)
    keywords = extract_keywords(extracted_text, top_n=20)

    if skip_ml:
        worthy_sentences = nlp_results["sentences"]
    else:
        from ml.predictor import filter_worthy_sentences
        worthy_sentences = filter_worthy_sentences(
            nlp_results["sentences"]
        )

    # Load saved questions per type from the database.
    saved_by_type = {}
    for quiz_type in ("MCQ", "FILL_BLANK", "TRUE_FALSE", "SHORT_ANSWER"):
        records = GeneratedQuestion.objects.filter(
            document=document_record,
            question_type=quiz_type,
        )
        saved_by_type[quiz_type] = _records_to_session_items(records)

    # Build the combined quiz from saved questions.
    saved_by_type["COMBINED"] = _build_combined_quiz(saved_by_type)

    # Store the quizzes in session (answer keys stay server-side).
    _store_quiz_session(
        request,
        document_record.id,
        saved_by_type,
    )

    fill_blank_questions = generate_fill_blank_questions(
        worthy_sentences, keywords, limit=10,
    )

    return {
        "document_id": document_record.id,
        "filename": document_record.original_filename,
        "page_count": document_record.page_count,
        "extracted_text": extracted_text,
        "sentences": nlp_results["sentences"],
        "processed_sentences": nlp_results["processed_sentences"],
        "keywords": keywords,
        "questions": fill_blank_questions,
        "mcq_questions": _public_items(saved_by_type["MCQ"]),
        "fitb_questions": _public_items(saved_by_type["FILL_BLANK"]),
        "tf_questions": _public_items(saved_by_type["TRUE_FALSE"]),
        "short_questions": _public_items(saved_by_type["SHORT_ANSWER"]),
        "combined_questions": _public_items(saved_by_type["COMBINED"]),
        "sentences_before_ml": len(nlp_results["sentences"]),
        "sentences_after_ml": len(worthy_sentences),
        "ml_skipped": skip_ml,
    }


def _score_submission(request, quiz_type, document_record):
    """
    Score the submitted form for the given quiz type.

    Returns a dict with quiz_results, score, total_questions,
    percentage and answers_to_save, or None on failure.
    """

    session_key = SESSION_KEYS[quiz_type]
    active_quiz = request.session.get(session_key, [])

    if not active_quiz:
        return None

    question_ids = [item["question_id"] for item in active_quiz]

    if quiz_type == "COMBINED":
        database_questions = GeneratedQuestion.objects.filter(
            document=document_record,
            id__in=question_ids,
        )
    else:
        database_questions = GeneratedQuestion.objects.filter(
            document=document_record,
            id__in=question_ids,
            question_type=quiz_type,
        )

    question_lookup = {
        question.id: question
        for question in database_questions
    }

    if (
        len(question_lookup) != len(active_quiz)
        or any(
            item["question_id"] not in question_lookup
            for item in active_quiz
        )
    ):
        return None

    score = 0
    total_questions = len(active_quiz)
    quiz_results = []
    answers_to_save = []

    for index, item in enumerate(active_quiz):
        question_record = question_lookup[item["question_id"]]
        submitted = request.POST.get(f"answer_{index}", "").strip()

        correct_answer = question_record.correct_answer

        if quiz_type == "COMBINED":
            item_type = item.get("question_type", "")
        else:
            item_type = quiz_type

        if item_type == "MCQ":
            if submitted not in item["options"]:
                submitted = ""
            is_correct = (
                submitted.casefold() == correct_answer.casefold()
            ) if submitted else False

        elif item_type == "TRUE_FALSE":
            if submitted not in ("True", "False"):
                submitted = ""
            is_correct = submitted == correct_answer

        elif item_type == "FILL_BLANK":
            is_correct = (
                answers_match(submitted, correct_answer)
                if submitted else False
            )

        elif item_type == "SHORT_ANSWER":
            is_correct = (
                short_answer_contains_keyword(submitted, correct_answer)
                if submitted else False
            )

        else:
            is_correct = False

        if is_correct:
            score += 1

        answers_to_save.append({
            "question": question_record,
            "selected_answer": submitted,
            "is_correct": is_correct,
        })

        quiz_results.append({
            "number": index + 1,
            "question": question_record.question_text,
            "question_type": item_type,
            "options": item.get("options", []),
            "selected_answer": submitted if submitted else "No answer",
            "correct_answer": correct_answer,
            "is_correct": is_correct,
        })

    percentage = round(
        (score / total_questions) * 100, 2
    ) if total_questions else 0

    return {
        "quiz_results": quiz_results,
        "score": score,
        "total_questions": total_questions,
        "percentage": percentage,
        "answers_to_save": answers_to_save,
    }


def _handle_submit(request, quiz_type, document_record):
    """Score and save a quiz submission. Returns updated context."""

    outcome = _score_submission(request, quiz_type, document_record)

    if outcome is None:
        _clear_quiz_session(request)
        return {
            "error": (
                "Some quiz questions could not be found. "
                "Please generate a new quiz."
            ),
            "documents": UploadedDocument.objects.all(),
            "attempts": QuizAttempt.objects.select_related(
                "document"
            ).all(),
        }

    with transaction.atomic():
        attempt = QuizAttempt.objects.create(
            document=document_record,
            quiz_type=quiz_type,
            score=outcome["score"],
            total_questions=outcome["total_questions"],
            percentage=outcome["percentage"],
            submitted_at=timezone.now(),
        )

        StudentAnswer.objects.bulk_create([
            StudentAnswer(
                attempt=attempt,
                question=answer["question"],
                selected_answer=answer["selected_answer"],
                is_correct=answer["is_correct"],
            )
            for answer in outcome["answers_to_save"]
        ])

    context = _build_document_context(request, document_record)
    context.update({
        "quiz_submitted": True,
        "quiz_type": quiz_type,
        "quiz_type_label": QUIZ_LABELS[quiz_type],
        "quiz_results": outcome["quiz_results"],
        "score": outcome["score"],
        "total_questions": outcome["total_questions"],
        "percentage": outcome["percentage"],
        "documents": UploadedDocument.objects.all(),
        "attempts": QuizAttempt.objects.select_related(
            "document"
        ).all(),
    })

    request.session.pop(SESSION_KEYS[quiz_type], None)

    return context


def home(request):
    context = {}

    # ------- View a saved attempt -------
    if (
        request.method == "POST"
        and request.POST.get("action") == "view_attempt"
    ):
        attempt_id = request.POST.get("attempt_id")

        attempt = QuizAttempt.objects.filter(pk=attempt_id).first()

        if attempt is None:
            context["error"] = (
                "That saved attempt could not be found. "
                "It may have been deleted."
            )
            context["documents"] = UploadedDocument.objects.all()
            context["attempts"] = QuizAttempt.objects.select_related(
                "document"
            ).all()
            return render(request, "quiz/home.html", context)

        answers = attempt.answers.select_related("question").all()

        quiz_results = []
        for index, answer in enumerate(answers, start=1):
            quiz_results.append({
                "number": index,
                "question": answer.question.question_text,
                "question_type": answer.question.question_type,
                "options": answer.question.options,
                "selected_answer": (
                    answer.selected_answer
                    if answer.selected_answer
                    else "No answer"
                ),
                "correct_answer": answer.question.correct_answer,
                "is_correct": answer.is_correct,
            })

        context.update({
            "quiz_submitted": True,
            "viewing_saved_attempt": True,
            "saved_attempt_id": attempt.id,
            "saved_attempt_date": attempt.submitted_at or attempt.started_at,
            "quiz_type": attempt.quiz_type,
            "quiz_type_label": QUIZ_LABELS.get(
                attempt.quiz_type, attempt.quiz_type
            ),
            "quiz_results": quiz_results,
            "score": attempt.score,
            "total_questions": attempt.total_questions,
            "percentage": attempt.percentage,
            "documents": UploadedDocument.objects.all(),
            "attempts": QuizAttempt.objects.select_related(
                "document"
            ).all(),
        })

        return render(request, "quiz/home.html", context)

    # ------- Quiz submissions (one action per quiz type) -------
    submit_action = request.POST.get("action")

    if request.method == "POST" and submit_action in (
        "submit_quiz",
        "submit_fitb",
        "submit_tf",
        "submit_short",
        "submit_combined",
    ):
        action_to_type = {
            "submit_quiz": "MCQ",
            "submit_fitb": "FILL_BLANK",
            "submit_tf": "TRUE_FALSE",
            "submit_short": "SHORT_ANSWER",
            "submit_combined": "COMBINED",
        }
        quiz_type = action_to_type[submit_action]

        document_id = request.session.get("active_quiz_document_id")

        if not document_id:
            context["error"] = (
                "There is no active quiz. Please upload a PDF "
                "document and generate questions first."
            )
            context["documents"] = UploadedDocument.objects.all()
            context["attempts"] = QuizAttempt.objects.select_related(
                "document"
            ).all()
            return render(request, "quiz/home.html", context)

        try:
            document_record = UploadedDocument.objects.get(pk=document_id)
        except UploadedDocument.DoesNotExist:
            _clear_quiz_session(request)
            context["error"] = (
                "The document for this quiz could not be found. "
                "Please upload the PDF again."
            )
            context["documents"] = UploadedDocument.objects.all()
            context["attempts"] = QuizAttempt.objects.select_related(
                "document"
            ).all()
            return render(request, "quiz/home.html", context)

        context.update(_handle_submit(request, quiz_type, document_record))
        return render(request, "quiz/home.html", context)

    # ------- PDF upload -------
    if request.method == "POST":
        uploaded_file = request.FILES.get("document")

        if not uploaded_file:
            context["error"] = "Please select a PDF document."

        elif not uploaded_file.name.lower().endswith(".pdf"):
            context["error"] = "Only PDF documents are supported."

        elif uploaded_file.size > MAX_FILE_SIZE:
            context["error"] = "The PDF must not exceed 10 MB."

        else:
            try:
                pdf_bytes = uploaded_file.read()

                if not pdf_bytes.startswith(b"%PDF-"):
                    raise ValueError("Invalid PDF file.")

                with pymupdf.open(stream=pdf_bytes, filetype="pdf") as pdf:
                    page_count = pdf.page_count

                    if page_count == 0:
                        raise ValueError("The PDF has no pages.")

                    page_texts = []

                    for page_number, page in enumerate(pdf, start=1):
                        page_text = page.get_text().strip()

                        if page_text:
                            page_texts.append(
                                f"Page {page_number}\n{page_text}"
                            )

                    extracted_text = "\n\n".join(page_texts)

                if not extracted_text.strip():
                    context["error"] = (
                        "No readable text was found. "
                        "This PDF may contain scanned images."
                    )

                else:
                    original_name = Path(uploaded_file.name).name
                    safe_name = f"{uuid.uuid4().hex}_{original_name}"
                    skip_ml = request.POST.get("skip_ml") == "1"

                    document_record = UploadedDocument(
                        original_filename=original_name,
                        page_count=page_count,
                        extracted_text=extracted_text,
                    )
                    document_record.file.save(
                        safe_name,
                        ContentFile(pdf_bytes),
                        save=False,
                    )
                    document_record.save()

                    nlp_results = process_text(extracted_text)
                    keywords = extract_keywords(extracted_text, top_n=20)

                    if skip_ml:
                        worthy_sentences = nlp_results["sentences"]
                    else:
                        from ml.predictor import filter_worthy_sentences
                        worthy_sentences = filter_worthy_sentences(
                            nlp_results["sentences"]
                        )

                    fill_blank_questions = generate_fill_blank_questions(
                        worthy_sentences, keywords, limit=10,
                    )
                    mcq_questions = generate_multiple_choice_questions(
                        worthy_sentences, keywords, limit=10,
                    )
                    tf_questions = generate_true_false_questions(
                        worthy_sentences, keywords, limit=10,
                    )
                    short_questions = generate_short_answer_questions(
                        worthy_sentences, keywords, limit=10,
                    )

                    for question in fill_blank_questions:
                        GeneratedQuestion.objects.create(
                            document=document_record,
                            question_text=question["question"],
                            question_type="FILL_BLANK",
                            options=[],
                            correct_answer=question["answer"],
                            source_sentence=question.get(
                                "source_sentence", ""
                            ),
                        )

                    for question in mcq_questions:
                        GeneratedQuestion.objects.create(
                            document=document_record,
                            question_text=question["question"],
                            question_type="MCQ",
                            options=question["options"],
                            correct_answer=question["answer"],
                            source_sentence=question.get(
                                "source_sentence", ""
                            ),
                        )

                    for question in tf_questions:
                        GeneratedQuestion.objects.create(
                            document=document_record,
                            question_text=question["question"],
                            question_type="TRUE_FALSE",
                            options=["True", "False"],
                            correct_answer=question["answer"],
                            source_sentence=question.get(
                                "source_sentence", ""
                            ),
                        )

                    for question in short_questions:
                        GeneratedQuestion.objects.create(
                            document=document_record,
                            question_text=question["question"],
                            question_type="SHORT_ANSWER",
                            options=[],
                            correct_answer=question["answer"],
                            source_sentence=question.get(
                                "source_sentence", ""
                            ),
                        )

                    saved_by_type = {}
                    for quiz_type in ("MCQ", "FILL_BLANK", "TRUE_FALSE", "SHORT_ANSWER"):
                        records = GeneratedQuestion.objects.filter(
                            document=document_record,
                            question_type=quiz_type,
                        )
                        saved_by_type[quiz_type] = (
                            _records_to_session_items(records)
                        )

                    saved_by_type["COMBINED"] = _build_combined_quiz(saved_by_type)

                    _store_quiz_session(
                        request,
                        document_record.id,
                        saved_by_type,
                    )

                    context.update({
                        "success": (
                            "PDF uploaded and processed successfully. "
                            "The document and generated questions "
                            "have been saved."
                        ),
                        "document_id": document_record.id,
                        "filename": original_name,
                        "page_count": page_count,
                        "extracted_text": extracted_text,
                        "sentences": nlp_results["sentences"],
                        "processed_sentences": (
                            nlp_results["processed_sentences"]
                        ),
                        "keywords": keywords,
                        "questions": fill_blank_questions,
                        "mcq_questions": _public_items(
                            saved_by_type["MCQ"]
                        ),
                        "fitb_questions": _public_items(
                            saved_by_type["FILL_BLANK"]
                        ),
                        "tf_questions": _public_items(
                            saved_by_type["TRUE_FALSE"]
                        ),
                        "short_questions": _public_items(
                            saved_by_type["SHORT_ANSWER"]
                        ),
                        "combined_questions": _public_items(
                            saved_by_type["COMBINED"]
                        ),
                        "sentences_before_ml": len(
                            nlp_results["sentences"]
                        ),
                        "sentences_after_ml": len(worthy_sentences),
                        "ml_skipped": skip_ml,
                    })

                    if not any(
                        [mcq_questions, fill_blank_questions,
                         tf_questions, short_questions]
                    ):
                        context["question_message"] = (
                            "The document was processed, but no suitable "
                            "questions could be generated. Try a longer "
                            "PDF containing readable academic text and "
                            "enough distinct keywords."
                        )

            except (
                ValueError,
                pymupdf.FileDataError,
                RuntimeError,
            ):
                context["error"] = (
                    "The PDF could not be read. "
                    "Please try another document."
                )

    context["documents"] = UploadedDocument.objects.all()
    context["attempts"] = QuizAttempt.objects.select_related(
        "document"
    ).all()
    return render(request, "quiz/home.html", context)


def load_document(request, document_id):
    """Load a previously uploaded document into all tabs."""

    document_record = get_object_or_404(UploadedDocument, pk=document_id)

    context = _build_document_context(request, document_record)
    context["documents"] = UploadedDocument.objects.all()
    context["attempts"] = QuizAttempt.objects.select_related(
        "document"
    ).all()

    return render(request, "quiz/home.html", context)


def delete_document(request, document_id):
    """Delete a document and all its questions and attempts."""

    if request.method != "POST":
        return redirect("home")

    document_record = get_object_or_404(UploadedDocument, pk=document_id)

    if request.session.get("active_quiz_document_id") == document_record.id:
        _clear_quiz_session(request)

    if document_record.file:
        document_record.file.delete(save=False)

    document_record.delete()

    return redirect("/#documents")