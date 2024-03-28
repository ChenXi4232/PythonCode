import json

from django.core import serializers
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_http_methods

from backend.models import Book


@require_http_methods(["GET"])
def add_book(request):
    response = {}
    try:
        book_name = request.GET.get("book_name")
        book = Book(bookName=book_name)
        book.save()
        response["respMsg"] = "success"
        response["respCode"] = "000000"
    except Exception as e:
        response["respMsg"] = str(e)
        response["respCode"] = "999999"
    return JsonResponse(response)


@require_http_methods(["GET"])
def show_books(request):
    response = {}
    try:
        books = Book.objects.filter()
        response["list"] = json.loads(serializers.serialize("json", books))
        response["respMsg"] = "success"
        response["respCode"] = "000000"
    except Exception as e:
        response["respMsg"] = str(e)
        response["respCode"] = "999999"
    return JsonResponse(response)
