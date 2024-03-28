import openpyxl
from datetime import datetime


def timer(start_time=None):  # 计时函数
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(),
                                 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('Time taken: %i hours %i minutes and %s seconds.' %
              (thour, tmin, round(tsec, 2)))


def check_result():  # 检查函数
    result = openpyxl.load_workbook('随机分组结果.xlsx')  # 导入分组结果
    data = openpyxl.load_workbook(
        'D:\\Py_program\\code\\random\\data1.xlsx')  # 导入原始数据
    sheet_physics = result['物理组']
    sheet_history = result['历史组']
    sheet_data = data['Sheet1']
    name_list = []  # 存储学生
    gender1 = [0 for i in range(0, 25)]  # 存储物理组每组女生人数
    rank = -1  # 排除表头
    for row1 in sheet_physics.values:  # 物理组统计每组女生人数并检查是否分组正确且均有分组
        rank += 1
        i = 0
        while (i < 25):
            sign = 0  # 标记
            for name in name_list:  # 查重
                if name == row1[i]:
                    print("分组错误！")
                    exit(0)
            if rank != 0:
                name_list.append(row1[i])
            for row2 in sheet_data.values:  # 筛查列表中是否有对应正确学生
                if row2[1] == row1[i]:
                    if row2[2] == '女':
                        gender1[i] += 1  # 统计女生人数
                    if type(row2[6]) == int:
                        sign = 1
            if sign == 0 and rank != 0:
                print("分组错误！")
                exit(0)
            i += 1
    gender2 = [0 for i in range(0, 15)]  # 存储历史组每组女生人数
    rank = -1  # 排除表头
    for row1 in sheet_history.values:  # 历史组统计每组女生人数并检查是否分组正确且均有分组
        rank += 1
        i = 0
        while (i < 15):
            sign = 0  # 标记
            for name in name_list:  # 查重
                if name == row1[i]:
                    print("分组错误！")
                    exit(0)
            if rank != 0:
                name_list.append(row1[i])
            for row2 in sheet_data.values:  # 筛查列表中是否有对应正确学生
                if row2[1] == row1[i]:
                    if row2[2] == '女':
                        gender2[i] += 1  # 统计女生人数
                    if type(row2[10]) == int:
                        sign = 1
            if sign == 0 and rank != 0:
                print("分组错误！")
                exit(0)
            i += 1
    i = 0
    j = 0
    while (i < 24):  # 判断物理组女生比例是否均匀
        j = i + 1
        while (j < 25):
            if abs(gender1[i] - gender1[j]) > 1:
                print("分组错误！")
                exit(0)
            j += 1
        i += 1
    i = 0
    j = 0
    while (i < 14):  # 判断历史组女生比例是否均匀
        j = i + 1
        while (j < 15):
            if abs(gender2[i] - gender2[j]) > 1:
                print("分组错误！")
                exit(0)
            j += 1
        i += 1
    if len(name_list) != 200:  # 查缺
        print("分组错误！")
        exit(0)
    print("分组正确！")


if __name__ == '__main__':
    start_time = timer(None)  # 开始计时
    check_result()
    timer(start_time)  # 停止计时
