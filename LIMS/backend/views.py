from django.shortcuts import render
from rest_framework import status
from rest_framework.response import Response
from rest_framework.decorators import api_view
from .models import Book, BookInfo, LibraryBranch, Reader
from .models import BorrowingInfo, ReservationInfo, LateFeeInfo
from .models import ReaderStudent, ReaderFaulty
from .models import Staff, LibraryDirector, Librarian, SecurityGuard, Janitor
from .serializers import LibraryBranchSerializer
from .serializers import BookInfoSerializer
from .serializers import BookSerializer
from .serializers import ReaderSerializer
from .serializers import ReaderStudentSerializer
from .serializers import ReaderFaultySerializer
from .serializers import BorrowingInfoSerializer
from .serializers import ReservationInfoSerializer
from .serializers import StaffSerializer
from .serializers import LibraryDirectorSerializer
from .serializers import LibrarianSerializer
from .serializers import SecurityGuardSerializer
from .serializers import JanitorSerializer
from .serializers import UpdateReaderSerializer
from .serializers import UpdateStaffSerializer
from .serializers import LateFeeInfoSerializer

from django.conf import settings
from django.db import transaction
import os
import datetime


@api_view(['POST'])
def create_library_branch(request):
    if request.method == 'POST':
        serializer = LibraryBranchSerializer(data=request.data)
        if serializer.is_valid():
            try:
                with transaction.atomic():
                    serializer.save()
                    return Response(serializer.data, status=status.HTTP_201_CREATED)
            except Exception as e:
                return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    return Response({"error": "Only POST method is allowed."}, status=status.HTTP_405_METHOD_NOT_ALLOWED)


@api_view(['POST'])
def create_book_info(request):
    if request.method == 'POST':
        serializer = BookInfoSerializer(data=request.data)
        if serializer.is_valid():
            try:
                with transaction.atomic():
                    serializer.save()
                    return Response(serializer.data, status=status.HTTP_201_CREATED)
            except Exception as e:
                return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    return Response({"error": "Only POST method is allowed."}, status=status.HTTP_405_METHOD_NOT_ALLOWED)


@api_view(['POST'])
def create_book(request):
    if request.method == 'POST':
        serializer = BookSerializer(data=request.data)
        if serializer.is_valid():
            try:
                with transaction.atomic():
                    serializer.save()
                    return Response(serializer.data, status=status.HTTP_201_CREATED)
            except Exception as e:
                return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    return Response({"error": "Only POST method is allowed."}, status=status.HTTP_405_METHOD_NOT_ALLOWED)


@api_view(['POST'])
def create_reader(request):
    if request.method == 'POST':
        serializer = ReaderSerializer(data=request.data)
        if serializer.is_valid():
            try:
                with transaction.atomic():
                    serializer.save()
                    return Response(serializer.data, status=status.HTTP_201_CREATED)
            except Exception as e:
                return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    return Response({"error": "Only POST method is allowed."}, status=status.HTTP_405_METHOD_NOT_ALLOWED)


@api_view(['POST'])
def create_reader_student(request):
    if request.method == 'POST':
        serializer = ReaderStudentSerializer(data=request.data)
        if serializer.is_valid():
            try:
                with transaction.atomic():
                    serializer.save()
                    return Response(serializer.data, status=status.HTTP_201_CREATED)
            except Exception as e:
                return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    return Response({"error": "Only POST method is allowed."}, status=status.HTTP_405_METHOD_NOT_ALLOWED)


@api_view(['POST'])
def create_reader_faulty(request):
    if request.method == 'POST':
        serializer = ReaderFaultySerializer(data=request.data)
        if serializer.is_valid():
            try:
                with transaction.atomic():
                    serializer.save()
                    return Response(serializer.data, status=status.HTTP_201_CREATED)
            except Exception as e:
                return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    return Response({"error": "Only POST method is allowed."}, status=status.HTTP_405_METHOD_NOT_ALLOWED)


@api_view(['POST'])
def create_borrowing_info(request):
    if request.method == 'POST':
        serializer = BorrowingInfoSerializer(data=request.data)
        if serializer.is_valid():
            book_id = serializer.validated_data['book_id']
            reader_card_id = serializer.validated_data['reader_card_id']
            try:
                # 检查读者证有效期，到期则无法借书
                reader = Reader.objects.get(pk=reader_card_id)
                if reader.expiration_date < datetime.date.today():
                    return Response({"error": "Reader card has expired."}, status=status.HTTP_400_BAD_REQUEST)
                if reader.borrowing_count >= reader.borrowing_limit:
                    return Response({"error": "Reader has reached borrowing limit."}, status=status.HTTP_400_BAD_REQUEST)
                book = Book.objects.get(pk=book_id)
                if book.status == 'B' or book.status == 'BR':
                    return Response({"error": "Book is already borrowed by others."}, status=status.HTTP_400_BAD_REQUEST)
                if book.status == 'R':
                    # 查询是否是预约读者且是否超过预约期限
                    reservationinfo = ReservationInfo.objects.filter(book_id=book_id, reader_card_id=reader_card_id, pickup_date=None)
                    if not reservationinfo.exists():
                        return Response({"error": "Book is already reserved by others."}, status=status.HTTP_400_BAD_REQUEST)
                    sign = False
                    for reservation in reservationinfo:
                        if reservation.reservation_date + datetime.timedelta(days=3) >= datetime.date.today():
                            sign = True
                            break
                    if not sign:
                        return Response({"error": "Your reservation have been overdue. Book is reserved by others and you cannot borrow it."}, status=status.HTTP_400_BAD_REQUEST)
                try:
                    with transaction.atomic():
                        serializer.save()
                        return Response(serializer.data, status=status.HTTP_201_CREATED)
                except Exception as e:
                    return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            # except Book.DoesNotExist:
            #     return Response({"error": "Book does not exist."}, status=status.HTTP_404_NOT_FOUND)
            # 书或读者不存在
            except Exception as e:
                return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    return Response({"error": "Only POST method is allowed."}, status=status.HTTP_405_METHOD_NOT_ALLOWED)


@api_view(['POST'])
def create_reservation_info(request):
    if request.method == 'POST':
        serializer = ReservationInfoSerializer(data=request.data)
        if serializer.is_valid():
            book_id = serializer.validated_data['book_id']
            reader_card_id = serializer.validated_data['reader_card_id']
            try:
                # 检查读者证有效期，到期则无法借书
                reader = Reader.objects.get(pk=reader_card_id)
                if reader.expiration_date < datetime.date.today():
                    return Response({"error": "Reader card has expired."}, status=status.HTTP_400_BAD_REQUEST)
                if reader.borrowing_count >= reader.borrowing_limit:
                    return Response({"error": "Reader has reached borrowing limit."}, status=status.HTTP_400_BAD_REQUEST)
                book = Book.objects.get(pk=book_id)
                # 检查书籍是否已借出
                if book.status == 'R' or book.status == 'BR':
                    return Response({"error": "Book is already reserved."}, status=status.HTTP_400_BAD_REQUEST)
                # 借书的人不能预约
                borrowinfo = BorrowingInfo.objects.filter(book_id=book_id, reader_card_id=reader_card_id, return_date=None)
                if borrowinfo.exists():
                    return Response({"error": "Book is already borrowed by you."}, status=status.HTTP_400_BAD_REQUEST)
                # 预约且三天内未取书的人从预约日算起十天内不得再次预约
                reservationinfo = ReservationInfo.objects.filter(book_id=book_id, reader_card_id=reader_card_id, pickup_date=None)
                if reservationinfo.exists():
                    sign = False
                    for reservation in reservationinfo:
                        if reservation.reservation_date + datetime.timedelta(days=3) < datetime.date.today():
                            if reservation.reservation_date + datetime.timedelta(days=10) >= datetime.date.today():
                                sign = True
                                break
                    if sign:
                        return Response({"error": "You have already reserved this book within past ten days but didn't take it, so you cannot reserve it again."}, status=status.HTTP_400_BAD_REQUEST)
                try:
                    with transaction.atomic():
                        # 创建预约
                        serializer.save()
                        return Response(serializer.data, status=status.HTTP_201_CREATED)
                except Exception as e:
                    return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            except Exception as e:
                return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    return Response({"error": "Only POST method is allowed."}, status=status.HTTP_405_METHOD_NOT_ALLOWED)


@api_view(['POST'])
def create_staff(request):
    if request.method == 'POST':
        serializer = StaffSerializer(data=request.data)
        if serializer.is_valid():
            try:
                with transaction.atomic():
                    serializer.save()
                    return Response(serializer.data, status=status.HTTP_201_CREATED)
            except Exception as e:
                return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    return Response({"error": "Only POST method is allowed."}, status=status.HTTP_405_METHOD_NOT_ALLOWED)


@api_view(['POST'])
def create_library_director(request):
    if request.method == 'POST':
        serializer = LibraryDirectorSerializer(data=request.data)
        if serializer.is_valid():
            try:
                with transaction.atomic():
                    serializer.save()
                    return Response(serializer.data, status=status.HTTP_201_CREATED)
            except Exception as e:
                return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    return Response({"error": "Only POST method is allowed."}, status=status.HTTP_405_METHOD_NOT_ALLOWED)


@api_view(['POST'])
def create_librarian(request):
    if request.method == 'POST':
        serializer = LibrarianSerializer(data=request.data)
        if serializer.is_valid():
            try:
                with transaction.atomic():
                    serializer.save()
                    return Response(serializer.data, status=status.HTTP_201_CREATED)
            except Exception as e:
                return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    return Response({"error": "Only POST method is allowed."}, status=status.HTTP_405_METHOD_NOT_ALLOWED)


@api_view(['POST'])
def create_security_guard(request):
    if request.method == 'POST':
        serializer = SecurityGuardSerializer(data=request.data)
        if serializer.is_valid():
            try:
                with transaction.atomic():
                    serializer.save()
                    return Response(serializer.data, status=status.HTTP_201_CREATED)
            except Exception as e:
                return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    return Response({"error": "Only POST method is allowed."}, status=status.HTTP_405_METHOD_NOT_ALLOWED)


@api_view(['POST'])
def create_janitor(request):
    if request.method == 'POST':
        serializer = JanitorSerializer(data=request.data)
        if serializer.is_valid():
            try:
                with transaction.atomic():
                    serializer.save()
                    return Response(serializer.data, status=status.HTTP_201_CREATED)
            except Exception as e:
                return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    return Response({"error": "Only POST method is allowed."}, status=status.HTTP_405_METHOD_NOT_ALLOWED)


# 续借
@api_view(['POST'])
def renew_book(request):
    if request.method == 'POST':
        serializer = BorrowingInfoSerializer(data=request.data)
        if serializer.is_valid():
            borrowing_id = serializer.validated_data['borrowing_id']
            try:
                borrowing_info = BorrowingInfo.objects.get(pk=borrowing_id)
                if borrowing_info.is_renewed:
                    return Response({"error": "Book has been renewed once."}, status=status.HTTP_400_BAD_REQUEST)
                borrowing_info.is_renewed = True
                borrowing_info.due_date += datetime.timedelta(days=30)
                try:
                    with transaction.atomic():
                        borrowing_info.save()
                        return Response(serializer.data, status=status.HTTP_200_OK)
                except Exception as e:
                    return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            except BorrowingInfo.DoesNotExist:
                return Response({"error": "Borrowing info does not exist."}, status=status.HTTP_404_NOT_FOUND)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    return Response({"error": "Only POST method is allowed."}, status=status.HTTP_405_METHOD_NOT_ALLOWED)


# 取消预约，即删除预约记录
@api_view(['POST'])
def cancel_reservation(request):
    if request.method == 'POST':
        serializer = ReservationInfoSerializer(data=request.data)
        if serializer.is_valid():
            reservation_id = serializer.validated_data['reservation_id']
            try:
                reservation_info = ReservationInfo.objects.get(pk=reservation_id)
                try:
                    with transaction.atomic():
                        reservation_info.delete()
                        return Response(serializer.data, status=status.HTTP_200_OK)
                except Exception as e:
                    return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            except ReservationInfo.DoesNotExist:
                return Response({"error": "Reservation info does not exist."}, status=status.HTTP_404_NOT_FOUND)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    return Response({"error": "Only POST method is allowed."}, status=status.HTTP_405_METHOD_NOT_ALLOWED)


# 还书
@api_view(['POST'])
def return_book(request):
    if request.method == 'POST':
        serializer = BorrowingInfoSerializer(data=request.data)
        if serializer.is_valid():
            borrowing_id = serializer.validated_data['borrowing_id']
            try:
                borrowing_info = BorrowingInfo.objects.get(pk=borrowing_id)
                if borrowing_info.return_date:
                    return Response({"error": "Book has been returned."}, status=status.HTTP_400_BAD_REQUEST)
                borrowing_info.return_date = datetime.date.today()
                try:
                    with transaction.atomic():
                        borrowing_info.save()
                        return Response(serializer.data, status=status.HTTP_200_OK)
                except Exception as e:
                    return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            except BorrowingInfo.DoesNotExist:
                return Response({"error": "Borrowing info does not exist."}, status=status.HTTP_404_NOT_FOUND)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    return Response({"error": "Only POST method is allowed."}, status=status.HTTP_405_METHOD_NOT_ALLOWED)


# 更新 Book 馆藏位置
@api_view(['POST'])
def update_book_location(request):
    if request.method == 'POST':
        serializer = BookSerializer(data=request.data)
        if serializer.is_valid():
            book_id = serializer.validated_data['book_id']
            try:
                book = Book.objects.get(pk=book_id)
                book.location = serializer.validated_data['location']
                try:
                    with transaction.atomic():
                        book.save()
                        return Response(serializer.data, status=status.HTTP_200_OK)
                except Exception as e:
                    return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            except Book.DoesNotExist:
                return Response({"error": "Book does not exist."}, status=status.HTTP_404_NOT_FOUND)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    return Response({"error": "Only POST method is allowed."}, status=status.HTTP_405_METHOD_NOT_ALLOWED)


# 删除 Book ，如果未还必须先设置还书
@api_view(['DELETE'])
def delete_book(request, book_id):
    try:
        book = Book.objects.get(pk=book_id)
        if book.status == 'B' or book.status == 'BR':
            return Response({"error": "Book is not returned yet."}, status=status.HTTP_400_BAD_REQUEST)
        try:
            with transaction.atomic():
                book.delete()
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    except Book.DoesNotExist:
        return Response({"error": "Book does not exist."}, status=status.HTTP_404_NOT_FOUND)


# 修改 BookInfo 信息
@api_view(['POST'])
def update_book_info(request):
    if request.method == 'POST':
        serializer = BookInfoSerializer(data=request.data)
        if serializer.is_valid():
            call_number = serializer.validated_data['call_number']
            try:
                book_info = BookInfo.objects.get(pk=call_number)
                if 'title' in serializer.validated_data:
                    book_info.title = serializer.validated_data['title']
                if 'publisher' in serializer.validated_data:
                    book_info.publisher = serializer.validated_data['publisher']
                if 'description' in serializer.validated_data:
                    book_info.description = serializer.validated_data['description']
                if 'isbn' in serializer.validated_data:
                    book_info.isbn = serializer.validated_data['isbn']
                if 'cover_image_path' in serializer.validated_data:
                    book_info.cover_image_path = serializer.validated_data['cover_image_path']
                if 'price' in serializer.validated_data:
                    book_info.price = serializer.validated_data['price']
                if 'category' in serializer.validated_data:
                    book_info.category = serializer.validated_data['category']
                if 'publication_date' in serializer.validated_data:
                    book_info.publication_date = serializer.validated_data['publication_date']
                if 'author' in serializer.validated_data:
                    book_info.author = serializer.validated_data['author']
                try:
                    with transaction.atomic():
                        book_info.save()
                        return Response(serializer.data, status=status.HTTP_200_OK)
                except Exception as e:
                    return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            except BookInfo.DoesNotExist:
                return Response({"error": "Book info does not exist."}, status=status.HTTP_404_NOT_FOUND)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    return Response({"error": "Only POST method is allowed."}, status=status.HTTP_405_METHOD_NOT_ALLOWED)


# 更新读者信息
@api_view(['POST'])
def update_reader(request):
    if request.method == 'POST':
        serializer = UpdateReaderSerializer(data=request.data)
        if serializer.is_valid():
            reader_card_id = serializer.validated_data['reader_card_id']
            try:
                reader = Reader.objects.get(pk=reader_card_id)
                if 'name' in serializer.validated_data:
                    reader.name = serializer.validated_data['name']
                if 'address' in serializer.validated_data:
                    reader.address = serializer.validated_data['address']
                if 'phone_number' in serializer.validated_data:
                    reader.phone_number = serializer.validated_data['phone_number']
                if 'email' in serializer.validated_data:
                    reader.email = serializer.validated_data['email']
                if 'gender' in serializer.validated_data:
                    reader.gender = serializer.validated_data['gender']
                if 'expiration_date' in serializer.validated_data:
                    reader.expiration_date = serializer.validated_data['expiration_date']
                if 'borrowing_limit' in serializer.validated_data:
                    reader.borrowing_limit = serializer.validated_data['borrowing_limit']
                if 'reader_card_photo' in serializer.validated_data:
                    reader.reader_card_photo = serializer.validated_data['reader_card_photo']
                if 'date_of_birth' in serializer.validated_data:
                    reader.date_of_birth = serializer.validated_data['date_of_birth']
                if 'outstanding_amount' in serializer.validated_data:
                    reader.outstanding_amount = serializer.validated_data['outstanding_amount']
                try:
                    with transaction.atomic():
                        reader.save()
                        return Response(serializer.data, status=status.HTTP_200_OK)
                except Exception as e:
                    return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            except Reader.DoesNotExist:
                return Response({"error": "Reader does not exist."}, status=status.HTTP_404_NOT_FOUND)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    return Response({"error": "Only POST method is allowed."}, status=status.HTTP_405_METHOD_NOT_ALLOWED)


# 删除 bookinfo
@api_view(['DELETE'])
def delete_book_info(request, call_number):
    try:
        book_info = BookInfo.objects.get(pk=call_number)
        # 如果 bookinfo 的书本总数不为0，那么查找所有 call_number 为 call_number 的 book,
        # 执行 book 的删除操作
        if book_info.total_quantity != 0:
            books = Book.objects.filter(call_number=call_number)
            for book in books:
                if book.status == 'B' or book.status == 'BR':
                    return Response({"error": "Book is not returned yet."}, status=status.HTTP_400_BAD_REQUEST)
                try:
                    with transaction.atomic():
                        book.delete()
                except Exception as e:
                    return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        try:
            with transaction.atomic():
                book_info.delete()
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    except BookInfo.DoesNotExist:
        return Response({"error": "Book info does not exist."}, status=status.HTTP_404_NOT_FOUND)


# 删除读者
@api_view(['DELETE'])
def delete_reader(request, reader_card_id):
    try:
        reader = Reader.objects.get(pk=reader_card_id)
        # 从 StudentReader 和 FaultyReader 中找到对应 reader_card_id 的记录并删除
        sign = False
        if not sign:
            try:
                reader_student = ReaderStudent.objects.get(pk=reader_card_id)
                sign = True
                try:
                    with transaction.atomic():
                        reader_student.delete()
                except Exception as e:
                    return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            except ReaderStudent.DoesNotExist:
                pass
        if not sign:
            try:
                reader_faulty = ReaderFaulty.objects.get(pk=reader_card_id)
                sign = True
                try:
                    with transaction.atomic():
                        reader_faulty.delete()
                except Exception as e:
                    return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            except ReaderFaulty.DoesNotExist:
                pass
        try:
            with transaction.atomic():
                reader.delete()
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    except Reader.DoesNotExist:
        return Response({"error": "Reader does not exist."}, status=status.HTTP_404_NOT_FOUND)


# 修改指定 reader_card_id 的学生读者的姓名或者学号
@api_view(['POST'])
def update_reader_student(request):
    if request.method == 'POST':
        serializer = ReaderStudentSerializer(data=request.data)
        if serializer.is_valid():
            reader_card_id = serializer.validated_data['reader_card_id']
            try:
                reader_student = ReaderStudent.objects.get(pk=reader_card_id)
                if 'major' in serializer.validated_data:
                    reader_student.major = serializer.validated_data['major']
                if 'student_id' in serializer.validated_data:
                    reader_student.student_id = serializer.validated_data['student_id']
                try:
                    with transaction.atomic():
                        reader_student.save()
                        return Response(serializer.data, status=status.HTTP_200_OK)
                except Exception as e:
                    return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            except ReaderStudent.DoesNotExist:
                return Response({"error": "Reader student does not exist."}, status=status.HTTP_404_NOT_FOUND)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    return Response({"error": "Only POST method is allowed."}, status=status.HTTP_405_METHOD_NOT_ALLOWED)


# 删除指定 reader_card_id 的学生读者
@api_view(['DELETE'])
def delete_reader_student(request, reader_card_id):
    try:
        reader_student = ReaderStudent.objects.get(pk=reader_card_id)
        try:
            with transaction.atomic():
                reader_student.delete()
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    except ReaderStudent.DoesNotExist:
        return Response({"error": "Reader student does not exist."}, status=status.HTTP_404_NOT_FOUND)


# 修改指定 reader_card_id 的教职工读者的部门或者工号
@api_view(['POST'])
def update_reader_faulty(request):
    if request.method == 'POST':
        serializer = ReaderFaultySerializer(data=request.data)
        if serializer.is_valid():
            reader_card_id = serializer.validated_data['reader_card_id']
            try:
                reader_faulty = ReaderFaulty.objects.get(pk=reader_card_id)
                if 'department' in serializer.validated_data:
                    reader_faulty.department = serializer.validated_data['department']
                if 'faculty_id' in serializer.validated_data:
                    reader_faulty.faculty_id = serializer.validated_data['faculty_id']
                try:
                    with transaction.atomic():
                        reader_faulty.save()
                        return Response(serializer.data, status=status.HTTP_200_OK)
                except Exception as e:
                    return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            except ReaderFaulty.DoesNotExist:
                return Response({"error": "Reader faulty does not exist."}, status=status.HTTP_404_NOT_FOUND)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    return Response({"error": "Only POST method is allowed."}, status=status.HTTP_405_METHOD_NOT_ALLOWED)


# 删除指定 reader_card_id 的教职工读者
@api_view(['DELETE'])
def delete_reader_faulty(request, reader_card_id):
    try:
        reader_faulty = ReaderFaulty.objects.get(pk=reader_card_id)
        try:
            with transaction.atomic():
                reader_faulty.delete()
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    except ReaderFaulty.DoesNotExist:
        return Response({"error": "Reader faulty does not exist."}, status=status.HTTP_404_NOT_FOUND)


# 修改 LateFeeInfo 的处理状态
@api_view(['POST'])
def update_late_fee_info(request):
    if request.method == 'POST':
        serializer = LateFeeInfoSerializer(data=request.data)
        if serializer.is_valid():
            late_fee_id = serializer.validated_data['late_fee_id']
            try:
                late_fee_info = LateFeeInfo.objects.get(pk=late_fee_id)
                late_fee_info.status = serializer.validated_data['status']
                try:
                    with transaction.atomic():
                        late_fee_info.save()
                        return Response(serializer.data, status=status.HTTP_200_OK)
                except Exception as e:
                    return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            except LateFeeInfo.DoesNotExist:
                return Response({"error": "Late fee info does not exist."}, status=status.HTTP_404_NOT_FOUND)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    return Response({"error": "Only POST method is allowed."}, status=status.HTTP_405_METHOD_NOT_ALLOWED)


# 删除图书馆分馆
@api_view(['DELETE'])
def delete_library_branch(request, branch_id):
    try:
        library_branch = LibraryBranch.objects.get(pk=branch_id)
        if library_branch.staff_number != 0:
            return Response({"error": "Library branch still has staff."}, status=status.HTTP_400_BAD_REQUEST)
        if library_branch.book_number != 0:
            return Response({"error": "Library branch still has books."}, status=status.HTTP_400_BAD_REQUEST)
        try:
            with transaction.atomic():
                library_branch.delete()
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    except LibraryBranch.DoesNotExist:
        return Response({"error": "Library branch does not exist."}, status=status.HTTP_404_NOT_FOUND)


# 更新 LibraryBranch 的信息
@api_view(['POST'])
def update_library_branch(request):
    if request.method == 'POST':
        serializer = LibraryBranchSerializer(data=request.data)
        if serializer.is_valid():
            branch_id = serializer.validated_data['branch_id']
            try:
                library_branch = LibraryBranch.objects.get(pk=branch_id)
                if 'branch_name' in serializer.validated_data:
                    library_branch.branch_name = serializer.validated_data['branch_name']
                if 'address' in serializer.validated_data:
                    library_branch.address = serializer.validated_data['address']
                if 'phone_number' in serializer.validated_data:
                    library_branch.phone_number = serializer.validated_data['phone_number']
                if 'email' in serializer.validated_data:
                    library_branch.email = serializer.validated_data['email']
                if 'opening_hours' in serializer.validated_data:
                    library_branch.opening_hours = serializer.validated_data['opening_hours']
                try:
                    with transaction.atomic():
                        library_branch.save()
                        return Response(serializer.data, status=status.HTTP_200_OK)
                except Exception as e:
                    return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            except LibraryBranch.DoesNotExist:
                return Response({"error": "Library branch does not exist."}, status=status.HTTP_404_NOT_FOUND)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    return Response({"error": "Only POST method is allowed."}, status=status.HTTP_405_METHOD_NOT_ALLOWED)


# 修改 Staff 的信息
@api_view(['POST'])
def update_staff(request):
    if request.method == 'POST':
        serializer = UpdateStaffSerializer(data=request.data)
        if serializer.is_valid():
            staff_id = serializer.validated_data['staff_id']
            try:
                staff = Staff.objects.get(pk=staff_id)
                if 'name' in serializer.validated_data:
                    staff.name = serializer.validated_data['name']
                if 'phone_number' in serializer.validated_data:
                    staff.phone_number = serializer.validated_data['phone_number']
                if 'email' in serializer.validated_data:
                    staff.email = serializer.validated_data['email']
                if 'gender' in serializer.validated_data:
                    staff.gender = serializer.validated_data['gender']
                if 'position' in serializer.validated_data:
                    staff.position = serializer.validated_data['position']
                if 'photo' in serializer.validated_data:
                    staff.photo = serializer.validated_data['photo']
                if 'birth_date' in serializer.validated_data:
                    staff.birth_date = serializer.validated_data['birth_date']
                if 'branch_id' in serializer.validated_data:
                    staff.branch_id = serializer.validated_data['branch_id']
                try:
                    with transaction.atomic():
                        staff.save()
                        return Response(serializer.data, status=status.HTTP_200_OK)
                except Exception as e:
                    return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            except Staff.DoesNotExist:
                return Response({"error": "Staff does not exist."}, status=status.HTTP_404_NOT_FOUND)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    return Response({"error": "Only POST method is allowed."}, status=status.HTTP_405_METHOD_NOT_ALLOWED)


# 修改 LibraryDirector 的信息
@api_view(['POST'])
def update_library_director(request):
    if request.method == 'POST':
        serializer = LibraryDirectorSerializer(data=request.data)
        if serializer.is_valid():
            staff_id = serializer.validated_data['staff_id']
            try:
                library_director = LibraryDirector.objects.get(pk=staff_id)
                if 'office_address' in serializer.validated_data:
                    library_director.office_address = serializer.validated_data['office_address']
                try:
                    with transaction.atomic():
                        library_director.save()
                        return Response(serializer.data, status=status.HTTP_200_OK)
                except Exception as e:
                    return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            except LibraryDirector.DoesNotExist:
                return Response({"error": "Library director does not exist."}, status=status.HTTP_404_NOT_FOUND)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    return Response({"error": "Only POST method is allowed."}, status=status.HTTP_405_METHOD_NOT_ALLOWED)


# 删除 LibraryDirector 的信息
@api_view(['DELETE'])
def delete_library_director(request, staff_id):
    try:
        library_director = LibraryDirector.objects.get(pk=staff_id)
        try:
            with transaction.atomic():
                library_director.delete()
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    except LibraryDirector.DoesNotExist:
        return Response({"error": "Library director does not exist."}, status=status.HTTP_404_NOT_FOUND)


# 修改 Librarian 的信息
@api_view(['POST'])
def update_librarian(request):
    if request.method == 'POST':
        serializer = LibrarianSerializer(data=request.data)
        if serializer.is_valid():
            staff_id = serializer.validated_data['staff_id']
            try:
                librarian = Librarian.objects.get(pk=staff_id)
                if "expertise" in serializer.validated_data:
                    librarian.expertise = serializer.validated_data['expertise']
                if "responsible_area" in serializer.validated_data:
                    librarian.responsible_area = serializer.validated_data['responsible_area']
                try:
                    with transaction.atomic():
                        librarian.save()
                        return Response(serializer.data, status=status.HTTP_200_OK)
                except Exception as e:
                    return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            except Librarian.DoesNotExist:
                return Response({"error": "Librarian does not exist."}, status=status.HTTP_404_NOT_FOUND)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    return Response({"error": "Only POST method is allowed."}, status=status.HTTP_405_METHOD_NOT_ALLOWED)


# 删除 Librarian 的信息
@api_view(['DELETE'])
def delete_librarian(request, staff_id):
    try:
        librarian = Librarian.objects.get(pk=staff_id)
        try:
            with transaction.atomic():
                librarian.delete()
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    except Librarian.DoesNotExist:
        return Response({"error": "Librarian does not exist."}, status=status.HTTP_404_NOT_FOUND)


# 修改 SecurityGuard 的信息
@api_view(['POST'])
def update_security_guard(request):
    if request.method == 'POST':
        serializer = SecurityGuardSerializer(data=request.data)
        if serializer.is_valid():
            staff_id = serializer.validated_data['staff_id']
            try:
                security_guard = SecurityGuard.objects.get(pk=staff_id)
                if "shift" in serializer.validated_data:
                    security_guard.shift = serializer.validated_data['shift']
                if "has_training_certificate" in serializer.validated_data:
                    security_guard.has_training_certificate = serializer.validated_data['phas_training_certificate']
                if "responsible_area" in serializer.validated_data:
                    security_guard.responsible_area = serializer.validated_data['responsible_area']
                try:
                    with transaction.atomic():
                        security_guard.save()
                        return Response(serializer.data, status=status.HTTP_200_OK)
                except Exception as e:
                    return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            except SecurityGuard.DoesNotExist:
                return Response({"error": "Security guard does not exist."}, status=status.HTTP_404_NOT_FOUND)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    return Response({"error": "Only POST method is allowed."}, status=status.HTTP_405_METHOD_NOT_ALLOWED)


# 删除 SecurityGuard 的信息
@api_view(['DELETE'])
def delete_security_guard(request, staff_id):
    try:
        security_guard = SecurityGuard.objects.get(pk=staff_id)
        try:
            with transaction.atomic():
                security_guard.delete()
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    except SecurityGuard.DoesNotExist:
        return Response({"error": "Security guard does not exist."}, status=status.HTTP_404_NOT_FOUND)


# 修改 Janitor 的信息
@api_view(['POST'])
def update_janitor(request):
    if request.method == 'POST':
        serializer = JanitorSerializer(data=request.data)
        if serializer.is_valid():
            staff_id = serializer.validated_data['staff_id']
            try:
                janitor = Janitor.objects.get(pk=staff_id)
                if "shift" in serializer.validated_data:
                    janitor.shift = serializer.validated_data['shift']
                if "responsible_area" in serializer.validated_data:
                    janitor.responsible_area = serializer.validated_data['responsible_area']
                try:
                    with transaction.atomic():
                        janitor.save()
                        return Response(serializer.data, status=status.HTTP_200_OK)
                except Exception as e:
                    return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            except Janitor.DoesNotExist:
                return Response({"error": "Janitor does not exist."}, status=status.HTTP_404_NOT_FOUND)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    return Response({"error": "Only POST method is allowed."}, status=status.HTTP_405_METHOD_NOT_ALLOWED)


# 删除 Janitor 的信息
@api_view(['DELETE'])
def delete_janitor(request, staff_id):
    try:
        janitor = Janitor.objects.get(pk=staff_id)
        try:
            with transaction.atomic():
                janitor.delete()
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    except Janitor.DoesNotExist:
        return Response({"error": "Janitor does not exist."}, status=status.HTTP_404_NOT_FOUND)


# 删除 Staff
@api_view(['DELETE'])
def delete_staff(request, staff_id):
    try:
        staff = Staff.objects.get(pk=staff_id)
        # 从 LibraryDirector, Librarian, SecurityGuard, Janitor 中找到对应 staff_id 的记录并删除
        sign = False
        if not sign:
            try:
                library_director = LibraryDirector.objects.get(pk=staff_id)
                sign = True
                try:
                    with transaction.atomic():
                        library_director.delete()
                except Exception as e:
                    return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            except LibraryDirector.DoesNotExist:
                pass
        if not sign:
            try:
                librarian = Librarian.objects.get(pk=staff_id)
                sign = True
                try:
                    with transaction.atomic():
                        librarian.delete()
                except Exception as e:
                    return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            except Librarian.DoesNotExist:
                pass
        if not sign:
            try:
                security_guard = SecurityGuard.objects.get(pk=staff_id)
                sign = True
                try:
                    with transaction.atomic():
                        security_guard.delete()
                except Exception as e:
                    return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            except SecurityGuard.DoesNotExist:
                pass
        if not sign:
            try:
                janitor = Janitor.objects.get(pk=staff_id)
                sign = True
                try:
                    with transaction.atomic():
                        janitor.delete()
                except Exception as e:
                    return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            except Janitor.DoesNotExist:
                pass
        try:
            with transaction.atomic():
                staff.delete()
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    except Staff.DoesNotExist:
        return Response({"error": "Staff does not exist."}, status=status.HTTP_404_NOT_FOUND)

