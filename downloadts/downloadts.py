import urllib
from concurrent.futures import ThreadPoolExecutor
import m3u8
data = m3u8.load("index0.m3u8").data

order_ts = []
# 将所有的带https的url存入order_ts
for i in data["segments"]:
    order_ts.append("play.modujx.com" + i["uri"])


def download(url, name):
    # 下载ts文件到D:\ProgramData\study\mov\tsfiles文件夹
    urllib.request.urlretrieve(
        url, 'D:\\study\\mov\\meimo\\'+name)


def read_name_url():
    name_url = {}
    for url in order_ts:
        name = url.split("/")[-1]
        name_url[name] = url
    return name_url


def download_tsfile():
    # 记录创立的线程
    task_list = []
    dict_name_url = read_name_url()
    # 线程池的创立
    pool = ThreadPoolExecutor(max_workers=50)
    for name in dict_name_url:
        # 启动多个线程下载文件，download是函数名，后面两个是参数值
        task_list.append(pool.submit(download, dict_name_url[name], name))
    # 判断所有下载线程是否全部结束
    while (True):
        if len(task_list) == 0:
            break
        for i in task_list:
            if i.done():
                task_list.remove(i)
                print("剩下任务数：{0}".format(len(task_list)))
    print("所有任务下载完成！")


download_tsfile()
