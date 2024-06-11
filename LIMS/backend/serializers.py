# serializers.py
import os
import datetime

from django.conf import settings
from rest_framework import serializers
from rest_framework import status
from rest_framework.response import Response
from .models import LibraryBranch, BookInfo, Book, Reader
from .models import ReaderStudent, ReaderFaulty
from .models import BorrowingInfo, ReservationInfo, LateFeeInfo
from .models import Staff
from .models import LibraryDirector, Librarian, SecurityGuard, Janitor


class LibraryBranchSerializer(serializers.ModelSerializer):
    class Meta:
        model = LibraryBranch
        fields = ['branch_id', 'branch_name', 'address', 'phone_number', 'email', 'opening_hours']
        read_only_fields = ['book_number', 'staff_number']


class BookInfoSerializer(serializers.ModelSerializer):
    cover_image = serializers.ImageField(required=False, write_only=True)

    class Meta:
        model = BookInfo
        fields = ['title', 'publisher', 'description', 'isbn', 'cover_image', 'price', 'category',
                  'publication_date', 'author', 'call_number']
        read_only_fields = ['total_quantity', 'borrowable_quantity', 'borrowing_count']

    def create(self, validated_data):
        cover_image = validated_data.pop('cover_image')
        book_info = BookInfo.objects.create(**validated_data)
        if cover_image:
            # 将图片保存到指定目录
            os.makedirs(os.path.join(settings.MEDIA_ROOT, 'figs/book'), exist_ok=True)
            image_path = os.path.join(settings.MEDIA_ROOT, 'figs/book', cover_image.name)
            with open(image_path, 'wb+') as destination:
                for chunk in cover_image.chunks():
                    destination.write(chunk)
            book_info.cover_image_path = os.path.join(settings.MEDIA_URL, 'figs/book', cover_image.name)
        else:
            return Response({"error": "Cover image is required."}, status=status.HTTP_400_BAD_REQUEST)
        return book_info


class BookSerializer(serializers.ModelSerializer):
    class Meta:
        model = Book
        fields = ['book_id', 'call_number', 'status', 'created_at', 'branch_location']
        read_only_fields = ['created_at']


class ReaderSerializer(serializers.ModelSerializer):
    reader_card_photo = serializers.ImageField(required=True, write_only=True)

    class Meta:
        model = Reader
        fields = ['name', 'address', 'phone_number', 'email', 'gender', 'reader_card_photo',
                  'reader_card_id', 'date_of_birth']
        read_only_fields = ['registration_date', 'expiration_data', 'borrowing_limit', 'outstanding_amount',
                            'borrowing_count']

    def create(self, validated_data):
        reader_card_photo = validated_data.pop('reader_card_photo')
        reader = Reader.objects.create(**validated_data)
        if reader_card_photo:
            os.makedirs(os.path.join(settings.MEDIA_ROOT, 'figs/reader'), exist_ok=True)
            image_path = os.path.join(settings.MEDIA_ROOT, 'figs/reader', reader_card_photo.name)
            with open(image_path, 'wb+') as destination:
                for chunk in reader_card_photo.chunks():
                    destination.write(chunk)
            reader.reader_card_photo = os.path.join(settings.MEDIA_URL, 'figs/reader', reader_card_photo.name)
        else:
            return Response({"error": "Reader card photo is required."}, status=status.HTTP_400_BAD_REQUEST)
        reader.expiration_date = reader.registration_date + datetime.timedelta(days=365 * 3)
        return reader


class ReaderStudentSerializer(serializers.ModelSerializer):
    class Meta:
        model = ReaderStudent
        fields = "__all__"


class ReaderFaultySerializer(serializers.ModelSerializer):
    class Meta:
        model = ReaderFaulty
        fields = "__all__"


class BorrowingInfoSerializer(serializers.ModelSerializer):
    class Meta:
        model = BorrowingInfo
        fields = ["book_id", "reader_card_id", "due_date"]
        read_only_fields = ["borrowing_id", "borrowing_date", "return_date", "is_renewed"]

    def create(self, validated_data):
        borrowing_info = BorrowingInfo.objects.create(**validated_data)
        borrowing_info.due_date = borrowing_info.borrowing_date + datetime.timedelta(days=60)
        return borrowing_info


class ReservationInfoSerializer(serializers.ModelSerializer):
    class Meta:
        model = ReservationInfo
        fields = ["book_id", "reader_card_id"]
        read_only_fields = ["reservation_id", "pickup_date", "reservation_date"]


class LateFeeInfoSerializer(serializers.ModelSerializer):
    class Meta:
        model = LateFeeInfo
        fields = ["late_fee_id", "status"]
        read_only_fields = ["reader_card_id", "book_id", "fine_amount", "late_days"]


class StaffSerializer(serializers.ModelSerializer):
    photo = serializers.ImageField(required=True, write_only=True)

    class Meta:
        model = Staff
        fields = ["name", "phone_number", "email", "gender", "position", "photo", "staff_id",
                  "birth_date", "branch_id"]
        read_only_fields = ["hire_date"]

    def create(self, validated_data):
        photo = validated_data.pop('photo')
        staff = Staff.objects.create(**validated_data)
        if photo:
            os.makedirs(os.path.join(settings.MEDIA_ROOT, 'figs/staff'), exist_ok=True)
            image_path = os.path.join(settings.MEDIA_ROOT, 'figs/staff', photo.name)
            with open(image_path, 'wb+') as destination:
                for chunk in photo.chunks():
                    destination.write(chunk)
            staff.photo_path = os.path.join(settings.MEDIA_URL, 'figs/staff', photo.name)
        else:
            return Response({"error": "Staff photo is required."}, status=status.HTTP_400_BAD_REQUEST)
        return staff


class LibraryDirectorSerializer(serializers.ModelSerializer):
    class Meta:
        model = LibraryDirector
        fields = "__all__"


class LibrarianSerializer(serializers.ModelSerializer):
    class Meta:
        model = Librarian
        fields = "__all__"


class SecurityGuardSerializer(serializers.ModelSerializer):
    class Meta:
        model = SecurityGuard
        fields = "__all__"


class JanitorSerializer(serializers.ModelSerializer):
    class Meta:
        model = Janitor
        fields = "__all__"


class UpdateReaderSerializer(serializers.ModelSerializer):
    reader_card_photo = serializers.ImageField(required=False, write_only=True)

    class Meta:
        model = Reader
        fields = ['name', 'address', 'phone_number', 'email', 'gender', 'reader_card_photo',
                  'reader_card_id', 'date_of_birth', 'expiration_data', 'borrowing_limit',
                  'outstanding_amount']
        read_only_fields = ['registration_date', 'borrowing_count']

    def create(self, validated_data):
        reader_card_photo = None
        if 'reader_card_photo' in validated_data:
            reader_card_photo = validated_data.pop('reader_card_photo')
        reader = Reader.objects.create(**validated_data)
        if reader_card_photo:
            os.makedirs(os.path.join(settings.MEDIA_ROOT, 'figs/reader'), exist_ok=True)
            image_path = os.path.join(settings.MEDIA_ROOT, 'figs/reader', reader_card_photo.name)
            with open(image_path, 'wb+') as destination:
                for chunk in reader_card_photo.chunks():
                    destination.write(chunk)
            reader.reader_card_photo = os.path.join(settings.MEDIA_URL, 'figs/reader', reader_card_photo.name)
        else:
            return Response({"error": "Reader card photo is required."}, status=status.HTTP_400_BAD_REQUEST)
        return reader


class UpdateStaffSerializer(serializers.ModelSerializer):
    photo = serializers.ImageField(required=False, write_only=True)

    class Meta:
        model = Staff
        fields = ["name", "phone_number", "email", "gender", "position", "photo", "staff_id",
                  "birth_date", "branch_id"]
        read_only_fields = ["hire_date"]

    def create(self, validated_data):
        photo = None
        if 'photo' in validated_data:
            photo = validated_data.pop('reader_card_photo')
        staff = Staff.objects.create(**validated_data)
        if photo:
            os.makedirs(os.path.join(settings.MEDIA_ROOT, 'figs/staff'), exist_ok=True)
            image_path = os.path.join(settings.MEDIA_ROOT, 'figs/staff', photo.name)
            with open(image_path, 'wb+') as destination:
                for chunk in photo.chunks():
                    destination.write(chunk)
            staff.photo = os.path.join(settings.MEDIA_URL, 'figs/staff', photo.name)
        else:
            return Response({"error": "Staff photo is required."}, status=status.HTTP_400_BAD_REQUEST)
        return staff
