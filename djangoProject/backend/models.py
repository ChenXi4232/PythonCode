from django.db import models


# Create your models here.


class Book(models.Model):
    # 如果没有指定主键的话Django会自动新增一个自增id作为主键
    objects = models.Manager()
    bookName = models.CharField(max_length=128, verbose_name="书名")
    createTime = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")

    def __unicode__(self):
        return self.bookName

    def __str__(self):
        return self.bookName
