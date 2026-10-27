from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("documents/<int:document_id>/", views.load_document, name="load_document"),
    path("documents/<int:document_id>/delete/", views.delete_document, name="delete_document"),
]