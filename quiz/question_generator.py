import random
import re
from difflib import SequenceMatcher


def _get_keyword_items(keywords):
    """
    Return valid, unique keyword records while preserving their
    extraction order and scores.
    """

    items = []
    seen = set()

    for item in keywords:
        if not isinstance(item, dict):
            continue

        word = item.get("word", "")
        score = item.get("score", 0)

        if not isinstance(word, str):
            continue

        word = word.strip()

        if not re.fullmatch(r"[A-Za-z]+", word):
            continue

        normalized = word.lower()

        if normalized in seen:
            continue

        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.0

        items.append({
            "word": word,
            "score": score,
        })
        seen.add(normalized)

    return items


def _get_keyword_words(keywords):
    """Return unique keyword words for compatibility with existing logic."""

    return [
        item["word"]
        for item in _get_keyword_items(keywords)
    ]


def _contains_word(sentence, word):
    """Check whether a complete word occurs in a sentence."""

    return bool(
        re.search(
            r"\b" + re.escape(word) + r"\b",
            sentence,
            re.IGNORECASE,
        )
    )


def _replace_first_word(sentence, word):
    """Replace the first occurrence of a word with a blank."""

    pattern = re.compile(
        r"\b" + re.escape(word) + r"\b",
        re.IGNORECASE,
    )

    return pattern.subn("________", sentence, count=1)


def _replace_first_word_with(sentence, old_word, new_word):
    """Replace the first occurrence of old_word with new_word."""

    pattern = re.compile(
        r"\b" + re.escape(old_word) + r"\b",
        re.IGNORECASE,
    )

    return pattern.subn(new_word, sentence, count=1)


def _similarity(first, second):
    """Measure spelling similarity between two words."""

    return SequenceMatcher(
        None,
        first.lower(),
        second.lower(),
    ).ratio()


def _normalize_answer(text):
    """Lowercase, strip, collapse whitespace, remove trailing punctuation."""

    if text is None:
        return ""

    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[.,;:!?]+$", "", text)
    return text


def answers_match(student_answer, correct_answer):
    """
    Case-insensitive exact match after normalising whitespace and
    trailing punctuation. Used for fill-in-the-blank scoring.
    """

    return _normalize_answer(student_answer) == _normalize_answer(
        correct_answer
    )


def short_answer_contains_keyword(student_answer, keyword):
    """
    True if the keyword appears as a complete word in the student
    answer, case-insensitive. Used for short-answer scoring.
    """

    normalized_answer = _normalize_answer(student_answer)
    normalized_keyword = _normalize_answer(keyword)

    if not normalized_answer or not normalized_keyword:
        return False

    return bool(
        re.search(
            r"\b" + re.escape(normalized_keyword) + r"\b",
            normalized_answer,
        )
    )


# ---------- Fill in the blank ----------

def generate_fill_blank_questions(sentences, keywords, limit=10):
    """
    Generate fill-in-the-blank questions using extracted keywords.

    Each answer must occur in its original source sentence.
    A keyword is used as a correct answer at most once.
    """

    questions = []
    used_answers = set()

    if limit <= 0:
        return questions

    keyword_items = _get_keyword_items(keywords)

    for sentence in sentences:
        sentence = sentence.strip()

        if len(sentence.split()) < 5:
            continue

        matching_keywords = [
            item
            for item in keyword_items
            if _contains_word(sentence, item["word"])
            and item["word"].lower() not in used_answers
        ]

        matching_keywords.sort(
            key=lambda item: (
                item["score"],
                len(item["word"]),
            ),
            reverse=True,
        )

        for item in matching_keywords:
            answer = item["word"]

            question_text, replacements = _replace_first_word(
                sentence,
                answer,
            )

            if replacements == 0:
                continue

            questions.append({
                "question": (
                    "Complete the sentence: " + question_text
                ),
                "answer": answer,
                "source_sentence": sentence,
                "type": "Fill in the blank",
            })

            used_answers.add(answer.lower())
            break

        if len(questions) >= limit:
            break

    return questions


# ---------- Multiple choice ----------

def _select_distractors(answer, sentence, keyword_items, count=3):
    """
    Select distinct distractors from the extracted document keywords.

    Distractors must not occur in the current sentence. Candidates
    with spelling very similar to the correct answer are excluded.
    Word length and keyword score are used to rank the remaining
    candidates.
    """

    answer_lower = answer.lower()
    candidates = []

    for position, item in enumerate(keyword_items):
        word = item["word"]
        word_lower = word.lower()

        if word_lower == answer_lower:
            continue

        if _contains_word(sentence, word):
            continue

        if _similarity(answer_lower, word_lower) >= 0.80:
            continue

        candidates.append({
            "word": word,
            "score": item["score"],
            "position": position,
        })

    answer_score = next(
        (
            keyword["score"]
            for keyword in keyword_items
            if keyword["word"].lower() == answer_lower
        ),
        0.0,
    )

    candidates.sort(
        key=lambda item: (
            abs(len(item["word"]) - len(answer)),
            abs(item["score"] - answer_score),
            item["position"],
        )
    )

    selected = []
    seen = {answer_lower}

    for candidate in candidates:
        word = candidate["word"]
        normalized = word.lower()

        if normalized in seen:
            continue

        selected.append(word)
        seen.add(normalized)

        if len(selected) == count:
            break

    return selected


def generate_multiple_choice_questions(
    sentences,
    keywords,
    limit=10,
):
    """
    Generate multiple-choice questions from document sentences.

    Each question contains four options: one extracted keyword answer
    and three distinct distractors not found in the source sentence.
    """

    questions = []
    used_answers = set()

    if limit <= 0:
        return questions

    keyword_items = _get_keyword_items(keywords)

    if len(keyword_items) < 4:
        return questions

    for sentence in sentences:
        sentence = sentence.strip()

        if len(sentence.split()) < 6:
            continue

        matching_keywords = [
            item
            for item in keyword_items
            if _contains_word(sentence, item["word"])
            and item["word"].lower() not in used_answers
        ]

        matching_keywords.sort(
            key=lambda item: (
                item["score"],
                len(item["word"]),
            ),
            reverse=True,
        )

        for item in matching_keywords:
            answer = item["word"]

            distractors = _select_distractors(
                answer=answer,
                sentence=sentence,
                keyword_items=keyword_items,
                count=3,
            )

            if len(distractors) != 3:
                continue

            question_text, replacements = _replace_first_word(
                sentence,
                answer,
            )

            if replacements == 0:
                continue

            options = [answer] + distractors
            random.shuffle(options)

            questions.append({
                "question": (
                    "Choose the correct word to complete the sentence: "
                    + question_text
                ),
                "options": options,
                "answer": answer,
                "source_sentence": sentence,
                "type": "Multiple choice",
            })

            used_answers.add(answer.lower())
            break

        if len(questions) >= limit:
            break

    return questions


# ---------- True / False ----------

def generate_true_false_questions(sentences, keywords, limit=10):
    """
    Generate balanced True/False questions.

    True statements are original sentences. False statements are the
    same sentence with one keyword replaced by another document keyword.

    Attempts to produce an equal number of True and False questions.
    Falls back to a smaller balanced set if enough material is not
    available. Never returns an unbalanced set.
    """

    if limit <= 0:
        return []

    keyword_items = _get_keyword_items(keywords)

    if len(keyword_items) < 2:
        return []

    # Aim for half True and half False.
    target_per_side = max(1, limit // 2)

    true_questions = []
    false_questions = []
    used_sentences = set()

    for sentence in sentences:
        sentence = sentence.strip()

        if len(sentence.split()) < 6:
            continue

        if sentence in used_sentences:
            continue

        matching_keywords = [
            item for item in keyword_items
            if _contains_word(sentence, item["word"])
        ]

        if not matching_keywords:
            continue

        matching_keywords.sort(
            key=lambda item: (item["score"], len(item["word"])),
            reverse=True,
        )

        keyword = matching_keywords[0]["word"]

        # Try to build a False version with a different keyword.
        replacement = None

        for candidate in keyword_items:
            candidate_word = candidate["word"]

            if candidate_word.lower() == keyword.lower():
                continue

            if _contains_word(sentence, candidate_word):
                continue

            if _similarity(keyword.lower(), candidate_word.lower()) >= 0.80:
                continue

            replacement = candidate_word
            break

        # Add a True statement if we still need one.
        if len(true_questions) < target_per_side:
            true_questions.append({
                "question": (
                    "True or False: " + sentence
                ),
                "options": ["True", "False"],
                "answer": "True",
                "source_sentence": sentence,
                "type": "True or False",
            })

        # Add a False statement if we still need one and can build it.
        if (
            replacement is not None
            and len(false_questions) < target_per_side
        ):
            corrupted, replacements = _replace_first_word_with(
                sentence,
                keyword,
                replacement,
            )

            if replacements > 0:
                false_questions.append({
                    "question": (
                        "True or False: " + corrupted
                    ),
                    "options": ["True", "False"],
                    "answer": "False",
                    "source_sentence": sentence,
                    "type": "True or False",
                })

        used_sentences.add(sentence)

        if (
            len(true_questions) >= target_per_side
            and len(false_questions) >= target_per_side
        ):
            break

    # Balance the two sides: never return an unbalanced set.
    balanced_count = min(
        len(true_questions),
        len(false_questions),
    )

    true_questions = true_questions[:balanced_count]
    false_questions = false_questions[:balanced_count]

    combined = true_questions + false_questions
    random.shuffle(combined)

    return combined


# ---------- Short answer ----------

def generate_short_answer_questions(sentences, keywords, limit=10):
    """
    Generate short-answer questions by blanking one keyword from a
    sentence and asking the student for the missing key term.

    The stored answer is the keyword itself; grading uses keyword
    containment (the student's typed answer must include the keyword
    as a complete word, case-insensitive).
    """

    questions = []
    used_answers = set()

    if limit <= 0:
        return questions

    keyword_items = _get_keyword_items(keywords)

    for sentence in sentences:
        sentence = sentence.strip()

        if len(sentence.split()) < 6:
            continue

        matching_keywords = [
            item
            for item in keyword_items
            if _contains_word(sentence, item["word"])
            and item["word"].lower() not in used_answers
        ]

        matching_keywords.sort(
            key=lambda item: (item["score"], len(item["word"])),
            reverse=True,
        )

        for item in matching_keywords:
            answer = item["word"]

            question_text, replacements = _replace_first_word(
                sentence,
                answer,
            )

            if replacements == 0:
                continue

            questions.append({
                "question": (
                    "What key term is missing from this sentence? "
                    + question_text
                ),
                "answer": answer,
                "source_sentence": sentence,
                "type": "Short answer",
            })

            used_answers.add(answer.lower())
            break

        if len(questions) >= limit:
            break

    return questions