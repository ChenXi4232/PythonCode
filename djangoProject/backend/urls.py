# -*- coding: utf-8 -*-

from django.urls import path

from backend import views

urlpatterns = [
    path("add_book", views.add_book),
    path("show_books", views.show_books),
]
