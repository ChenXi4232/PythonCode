from django.db import models
from django.conf import settings
import os


# Create your models here.
class Book(models.Model):
    BOOK_STATUS_CHOICES = (
        ('A', '可借阅'),
        ('B', '已借阅'),
        ('R', '已预约'),
        ('BR', '已借阅且已预约')
    )

    book_id = models.IntegerField(primary_key=True, verbose_name="图书 ID")
    call_number = models.ForeignKey('BookInfo', on_delete=models.CASCADE, verbose_name="索书号")
    status = models.CharField(max_length=10, choices=BOOK_STATUS_CHOICES, verbose_name="借阅状态")
    # 入库时间
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="入库时间")
    # 馆藏位置
    branch_location = models.ForeignKey('LibraryBranch', on_delete=models.CASCADE, verbose_name="馆藏位置")

    class Meta:
        unique_together = ('book_id', 'call_number')
        db_table = 'Book'
        verbose_name = '图书'
        verbose_name_plural = '图书'


class BookInfo(models.Model):
    title = models.CharField(max_length=255, verbose_name="书名")
    publisher = models.CharField(max_length=255, verbose_name="出版社")
    description = models.TextField(verbose_name="简介")
    isbn = models.CharField(max_length=20, verbose_name="ISBN")
    default_image = os.path.join(settings.MEDIA_URL, 'figs/default/book_default.jpg')
    cover_image_path = models.CharField(max_length=255, default=default_image, verbose_name="图书封面")
    price = models.DecimalField(max_digits=7, decimal_places=2, verbose_name="价格")
    category = models.CharField(max_length=100, verbose_name="类别")
    publication_date = models.DateField(verbose_name="出版日期")
    author = models.CharField(max_length=255, verbose_name="作者")
    # 索书号
    call_number = models.CharField(max_length=50, verbose_name="索书号", primary_key=True)
    # 现有总数
    total_quantity = models.IntegerField(verbose_name="现有总数", default=0)
    # 可借阅总数
    borrowable_quantity = models.IntegerField(verbose_name="可借阅总数", default=0)
    # 借阅总次数
    borrowing_count = models.IntegerField(verbose_name="借阅总次数", default=0)

    class Meta:
        db_table = 'BookInfo'
        verbose_name = '图书信息'
        verbose_name_plural = '图书信息'

    def __str__(self):
        return self.title


class Reader(models.Model):
    name = models.CharField(max_length=255, verbose_name="姓名")
    address = models.CharField(max_length=255, verbose_name="地址")
    phone_number = models.CharField(max_length=20, verbose_name="联系电话")
    email = models.EmailField(max_length=100, verbose_name="电子邮件")
    GENDER_CHOICES = [
        ('M', '男'),
        ('F', '女'),
        ('N', '未知')
    ]
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES, verbose_name="性别")
    registration_date = models.DateField(auto_now_add=True, verbose_name="注册日期")
    expiration_date = models.DateField(verbose_name="有效期")
    borrowing_limit = models.IntegerField(verbose_name="借阅限制", default=10)
    reader_card_photo = models.CharField(max_length=255, verbose_name="读者证照片")
    reader_card_id = models.CharField(max_length=50, primary_key=True, verbose_name="读者证 ID")
    date_of_birth = models.DateField(verbose_name="出生日期")
    outstanding_amount = models.DecimalField(max_digits=5, decimal_places=2, default=0, verbose_name="欠款金额")
    # 在借数目
    borrowing_count = models.IntegerField(default=0, verbose_name="在借数目")

    class Meta:
        db_table = "Reader"
        verbose_name = "读者"
        verbose_name_plural = "读者"

    def __str__(self):
        return self.name


class ReaderStudent(models.Model):
    reader_card_id = models.OneToOneField(Reader, on_delete=models.CASCADE, verbose_name="读者证 ID", primary_key=True)
    major = models.CharField(max_length=100, verbose_name="专业")
    student_id = models.CharField(max_length=50, verbose_name="学号")

    class Meta:
        db_table = "ReaderStudent"
        verbose_name = "学生"
        verbose_name_plural = "学生"

    def __str__(self):
        return self.student_id


class ReaderFaulty(models.Model):
    reader_card_id = models.OneToOneField(Reader, on_delete=models.CASCADE, verbose_name="读者证 ID", primary_key=True)
    department = models.CharField(max_length=100, verbose_name="部门")
    faculty_id = models.CharField(max_length=50, verbose_name="教职工工号")

    class Meta:
        db_table = "ReaderFaulty"
        verbose_name = "教职工"
        verbose_name_plural = "教职工"

    def __str__(self):
        return self.faculty_id


class BorrowingInfo(models.Model):
    borrowing_id = models.AutoField(primary_key=True, verbose_name="借阅 ID")
    borrowing_date = models.DateField(auto_now_add=True, verbose_name="借阅日期")
    due_date = models.DateField(verbose_name="应还日期")
    return_date = models.DateField(verbose_name="实际还书日期", blank=True, null=True)
    reader_card_id = models.CharField(max_length=50, verbose_name="读者证 ID")
    book_id = models.CharField(max_length=50, verbose_name="图书 ID")
    is_renewed = models.BooleanField(verbose_name="是否已续借", default=False)

    class Meta:
        db_table = "BorrowingInfo"
        verbose_name = "借阅信息"
        verbose_name_plural = "借阅信息"


class ReservationInfo(models.Model):
    reservation_id = models.AutoField(primary_key=True, verbose_name="预定 ID")
    reservation_date = models.DateField(auto_now_add=True, verbose_name="预定日期")
    pickup_date = models.DateField(verbose_name="取书日期", blank=True, null=True)
    reader_card_id = models.CharField(max_length=50, verbose_name="读者证 ID")
    book_id = models.CharField(max_length=50, verbose_name="图书 ID")

    class Meta:
        db_table = "ReservationInfo"
        verbose_name = "预定信息"
        verbose_name_plural = "预定信息"


class LateFeeInfo(models.Model):
    late_fee_id = models.AutoField(primary_key=True, verbose_name="违期 ID")
    late_days = models.IntegerField(verbose_name="违期天数")
    STATUS_CHOICES = [
        ('Pending', '待处理'),
        ('Paid', '已支付'),
        ('Waived', '已免除')
    ]
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default="Pending", verbose_name="处理状态")
    fine_amount = models.DecimalField(max_digits=5, decimal_places=2, verbose_name="罚款金额")
    reader_card_id = models.CharField(max_length=50, verbose_name="读者证 ID")
    book_id = models.CharField(max_length=50, verbose_name="图书 ID")

    class Meta:
        db_table = "LateFeeInfo"
        verbose_name = "违期信息"
        verbose_name_plural = "违期信息"


class LibraryBranch(models.Model):
    branch_id = models.CharField(max_length=50, primary_key=True, verbose_name="分馆 ID")
    branch_name = models.CharField(max_length=255, verbose_name="分馆名称")
    address = models.CharField(max_length=255, verbose_name="地址")
    phone_number = models.CharField(max_length=20, verbose_name="联系电话")
    email = models.EmailField(max_length=100, verbose_name="电子邮件")
    book_number = models.IntegerField(default=0, verbose_name="藏书数量")
    opening_hours = models.CharField(max_length=255, verbose_name="开馆时间")
    staff_number = models.IntegerField(default=0, verbose_name="员工数量")

    class Meta:
        db_table = "LibraryBranch"
        verbose_name = "图书馆分馆"
        verbose_name_plural = "图书馆分馆"


class Staff(models.Model):
    name = models.CharField(max_length=255, verbose_name="姓名")
    phone_number = models.CharField(max_length=20, verbose_name="联系电话")
    email = models.EmailField(max_length=100, verbose_name="电子邮件")
    GENDER_CHOICES = [
        ('M', '男'),
        ('F', '女'),
        ('N', '未知')
    ]
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES, verbose_name="性别")
    position = models.CharField(max_length=100, verbose_name="职位")
    photo_path = models.CharField(max_length=255, verbose_name="员工照片")
    Staff_id = models.CharField(max_length=50, primary_key=True, verbose_name="员工 ID")
    hire_date = models.DateField(auto_now_add=True, verbose_name="入职日期")
    birth_date = models.DateField(verbose_name="出生日期")
    # 负责分馆
    branch_id = models.ForeignKey(LibraryBranch, on_delete=models.CASCADE, verbose_name="分馆 ID")

    class Meta:
        db_table = "Staff"
        verbose_name = "员工"
        verbose_name_plural = "员工"

    def __str__(self):
        return self.name


class LibraryDirector(models.Model):
    office_address = models.CharField(max_length=255, verbose_name="办公室地址")
    Staff_id = models.OneToOneField(Staff, on_delete=models.CASCADE, verbose_name="员工 ID", primary_key=True)

    class Meta:
        db_table = "LibraryDirector"
        verbose_name = "图书馆馆长"
        verbose_name_plural = "图书馆馆长"


class Librarian(models.Model):
    expertise = models.CharField(max_length=100, verbose_name="专业领域")
    responsible_area = models.CharField(max_length=255, verbose_name="负责区域")
    Staff_id = models.OneToOneField(Staff, on_delete=models.CASCADE, verbose_name="员工 ID", primary_key=True)

    class Meta:
        db_table = "Librarian"
        verbose_name = "图书管理员"
        verbose_name_plural = "图书管理员"


class Janitor(models.Model):
    shift = models.CharField(max_length=50, verbose_name="班次")
    responsible_area = models.CharField(max_length=255, verbose_name="负责区域")
    Staff_id = models.OneToOneField(Staff, on_delete=models.CASCADE, verbose_name="员工 ID", primary_key=True)

    class Meta:
        db_table = "Janitor"
        verbose_name = "图书馆清洁人员"
        verbose_name_plural = "图书馆清洁人员"


class SecurityGuard(models.Model):
    has_training_certificate = models.BooleanField(verbose_name="安全培训证书")
    responsible_area = models.CharField(max_length=255, verbose_name="负责区域")
    shift = models.CharField(max_length=50, verbose_name="班次")
    Staff_id = models.OneToOneField(Staff, on_delete=models.CASCADE, verbose_name="员工 ID", primary_key=True)

    class Meta:
        db_table = "SecurityGuard"
        verbose_name = "图书馆安保人员"
        verbose_name_plural = "图书馆安保人员"
