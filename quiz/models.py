from django.db import models


class UploadedDocument(models.Model):
    original_filename = models.CharField(max_length=255)
    file = models.FileField(upload_to="documents/")
    page_count = models.PositiveIntegerField(default=0)
    extracted_text = models.TextField()
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.original_filename

    class Meta:
        ordering = ["-uploaded_at"]
        verbose_name = "Uploaded Document"
        verbose_name_plural = "Uploaded Documents"


class GeneratedQuestion(models.Model):
    QUESTION_TYPES = [
        ("MCQ", "Multiple Choice"),
        ("FILL_BLANK", "Fill in the Blank"),
        ("TRUE_FALSE", "True or False"),
        ("SHORT_ANSWER", "Short Answer"),
    ]

    document = models.ForeignKey(
        UploadedDocument,
        on_delete=models.CASCADE,
        related_name="questions",
    )
    question_text = models.TextField()
    question_type = models.CharField(
        max_length=20,
        choices=QUESTION_TYPES,
    )
    options = models.JSONField(default=list, blank=True)
    correct_answer = models.TextField()
    source_sentence = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.get_question_type_display()}: {self.question_text[:80]}"

    class Meta:
        ordering = ["created_at"]
        verbose_name = "Generated Question"
        verbose_name_plural = "Generated Questions"


class QuizAttempt(models.Model):
    QUIZ_TYPES = [
        ("MCQ", "Multiple Choice"),
        ("FILL_BLANK", "Fill in the Blank"),
        ("TRUE_FALSE", "True or False"),
        ("SHORT_ANSWER", "Short Answer"),
        ("COMBINED", "Combined"),
    ]

    document = models.ForeignKey(
        UploadedDocument,
        on_delete=models.CASCADE,
        related_name="attempts",
    )
    quiz_type = models.CharField(
        max_length=20,
        choices=QUIZ_TYPES,
        default="MCQ",
    )
    student_name = models.CharField(max_length=100, blank=True)
    score = models.PositiveIntegerField(default=0)
    total_questions = models.PositiveIntegerField(default=0)
    percentage = models.DecimalField(
        max_digits=5,
        decimal_places=2,
        default=0,
    )
    started_at = models.DateTimeField(auto_now_add=True)
    submitted_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return (
            f"{self.get_quiz_type_display()} Attempt {self.pk} - "
            f"{self.score}/{self.total_questions}"
        )

    class Meta:
        ordering = ["-started_at"]
        verbose_name = "Quiz Attempt"
        verbose_name_plural = "Quiz Attempts"


class StudentAnswer(models.Model):
    attempt = models.ForeignKey(
        QuizAttempt,
        on_delete=models.CASCADE,
        related_name="answers",
    )
    question = models.ForeignKey(
        GeneratedQuestion,
        on_delete=models.CASCADE,
        related_name="student_answers",
    )
    selected_answer = models.TextField(blank=True)
    is_correct = models.BooleanField(default=False)

    def __str__(self):
        return (
            f"Answer for question {self.question_id} "
            f"in attempt {self.attempt_id}"
        )

    class Meta:
        verbose_name = "Student Answer"
        verbose_name_plural = "Student Answers"