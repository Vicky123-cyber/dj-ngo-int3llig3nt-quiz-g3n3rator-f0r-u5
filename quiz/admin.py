
from django.contrib import admin

from .models import (
    UploadedDocument,
    GeneratedQuestion,
    QuizAttempt,
    StudentAnswer,
)


@admin.register(UploadedDocument)
class UploadedDocumentAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "original_filename",
        "page_count",
        "uploaded_at",
    )
    search_fields = ("original_filename",)
    readonly_fields = ("uploaded_at",)


@admin.register(GeneratedQuestion)
class GeneratedQuestionAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "question_type",
        "document",
        "created_at",
    )
    list_filter = ("question_type", "created_at")
    search_fields = (
        "question_text",
        "correct_answer",
        "document__original_filename",
    )
    readonly_fields = ("created_at",)


@admin.register(QuizAttempt)
class QuizAttemptAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "student_name",
        "document",
        "score",
        "total_questions",
        "percentage",
        "started_at",
        "submitted_at",
    )
    list_filter = ("started_at", "submitted_at")
    search_fields = (
        "student_name",
        "document__original_filename",
    )
    readonly_fields = ("started_at",)


@admin.register(StudentAnswer)
class StudentAnswerAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "attempt",
        "question",
        "is_correct",
    )
    list_filter = ("is_correct",)
    search_fields = (
        "selected_answer",
        "question__question_text",
    )
