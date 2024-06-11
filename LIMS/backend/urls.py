from django.urls import path
from . import views

urlpatterns = [
    path('library-branch/', views.create_library_branch, name='create_library_branch'),
    path('book-info/', views.create_book_info, name='create_book_info'),
    path('book/', views.create_book, name='create_book'),
    path('reader/', views.create_reader, name='create_reader'),
    path('reader-student/', views.create_reader_student, name='create_reader_student'),
    path('reader-faulty/', views.create_reader_faulty, name='create_reader_faulty'),
    path('borrowing-info/', views.create_borrowing_info, name='create_borrowing_info'),
    path('reservation-info/', views.create_reservation_info, name='create_reservation_info'),
    path('staff/', views.create_staff, name='create_staff'),
    path('library-director/', views.create_library_director, name='create_library_director'),
    path('librarian/', views.create_librarian, name='create_librarian'),
    path('security-guard/', views.create_security_guard, name='create_security_guard'),
    path('janitor/', views.create_janitor, name='create_janitor'),
]
