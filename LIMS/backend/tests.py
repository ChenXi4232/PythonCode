# tests.py
from django.test import TestCase, Client
from django.urls import reverse
from .models import BookInfo
import os
from django.conf import settings


class BookInfoTransactionTestCase(TestCase):
    def setUp(self):
        self.client = Client()
        self.url = reverse('create_book_info')
        self.test_image_path = os.path.join(settings.MEDIA_ROOT, 'test_image.jpeg')

        # 创建一个测试图片文件
        # with open(self.test_image_path, 'wb') as f:
        #     f.write(os.urandom(1024))  # 创建1KB的随机字节文件

    def tearDown(self):
        # 删除测试图片文件
        # if os.path.exists(self.test_image_path):
        #     os.remove(self.test_image_path)

        # 删除所有上传的测试文件
        # if os.path.exists(settings.MEDIA_ROOT):
        #     for root, dirs, files in os.walk(settings.MEDIA_ROOT):
        #         for file in files:
        #             os.remove(os.path.join(root, file))
        return

    # def test_transaction_rollback_on_error(self):
    #     # 构造一个无效的payload，缺少"call_number"字段
    #     payload = {
    #         'title': 'Test Book',
    #         'publisher': 'Test Publisher',
    #         'description': 'Test Description',
    #         'isbn': '1234567890',
    #         'cover_image': open(self.test_image_path, 'rb'),
    #         'price': '29.99',
    #         'category': 'Test Category',
    #         'publication_date': '2023-01-01',
    #         'author': 'Test Author',
    #         'total_quantity': 10,
    #         'borrowable_quantity': 5,
    #     }
    #
    #     response = self.client.post(self.url, data=payload)
    #
    #     # 检查响应状态码
    #     self.assertEqual(response.status_code, 400)
    #
    #     # 验证数据库中没有创建任何BookInfo对象
    #     self.assertEqual(BookInfo.objects.count(), 0)

    def test_successful_book_info_creation(self):
        # 构造一个有效的payload
        payload = {
            'title': 'Test Book',
            'publisher': 'Test Publisher',
            'description': 'Test Description',
            'isbn': '1234567890',
            'cover_image': open(self.test_image_path, 'rb'),
            'price': '29.99',
            'category': 'Test Category',
            'publication_date': '2023-01-01',
            'author': 'Test Author',
            'call_number': 'TEST12345',
            'total_quantity': 10,
            'borrowable_quantity': 5,
        }

        response = self.client.post(self.url, data=payload)
        print(response.content)

        # 检查响应状态码
        self.assertEqual(response.status_code, 201)

        # 验证数据库中创建了一个BookInfo对象
        self.assertEqual(BookInfo.objects.count(), 1)
        book_info = BookInfo.objects.first()
        self.assertEqual(book_info.title, 'Test Book')
        self.assertTrue(os.path.exists(os.path.join(settings.MEDIA_ROOT, 'figs/book', 'test_image.jpeg')))
