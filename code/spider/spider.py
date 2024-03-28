import requests
import pandas as pd
from bs4 import BeautifulSoup
import pymongo

df_book = pd.read_csv('Book_id.csv')
df_movie = pd.read_csv('Movie_id.csv')

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36 QIHU 360SE'}
conn = pymongo.MongoClient()
db = conn['douban']
db_book = db['book']
num = 1200


def spider(i):
    book_url = "https://book.douban.com/subject/" + \
        str(df_book.loc[i, "head"])+"/"
    book_html_ = requests.get(url=book_url, headers=headers)
    book_html = book_html_.text
    book_soup = BeautifulSoup(book_html, 'lxml')
    book = {}

    try:
        book_rank_text = book_soup.find("span", class_="rank-label-link").text
        book['排名'] = book_rank_text.strip()
    except:
        pass

    try:
        book_name_text = book_soup.find('h1').text
        book['书名'] = book_name_text.strip()
    except:
        # 将print出来的内容写入txt文件
        with open('error.txt', 'a') as f:
            f.write(str(df_book.loc[i, "head"])+'\n')

    try:
        book_img_text = book_soup.find(
            "div", id="mainpic").find('a').attrs['href']
        book['图片'] = book_img_text.strip()
    except:
        pass

    try:
        book_info_text = book_soup.find('div', id="info").text
        tmp = book_info_text.split('\n')
        tmp2, tmp3 = [], []
        for item in tmp:
            temp = item.split(':')
            tmp2.extend(temp)
        for item in tmp2:
            item = item.strip()
            if item:
                tmp3.append(item)
        title_lst_prev = book_soup.find(
            'div', id="info").findAll('span', class_="pl")
        title_lst = []
        for item in title_lst_prev:
            title_lst.append(item.text.strip().replace(":", ""))
        k, j = 1, 1
        txt = ''
        while j < len(tmp3) and k < len(title_lst):
            if tmp3[j] != title_lst[k]:
                txt += tmp3[j]
                j += 1
            else:
                book[title_lst[k-1]] = txt
                txt = ''
                j += 1
                k += 1
        while j < len(tmp3):
            txt += tmp3[j]
            j += 1
        book[title_lst[k-1]] = txt
    except:
        pass

    try:
        book_rating_text = book_soup.find('strong').text
        book_rating_num_text = book_soup.find('span', property="v:votes").text
        book['评分'] = book_rating_text.strip()
        book['评分人数'] = book_rating_num_text.strip()
    except:
        pass

    try:
        lst = book_soup.find('div', class_='related_info').findAll('h2')
        for item in lst:
            try:
                tmp = item.find('span').text
                if tmp == '内容简介':
                    content_intro = item.find_next_sibling().text.strip()
                    book['内容简介'] = content_intro
                    if '\n' in content_intro:
                        content_lst = content_intro.split('\n')
                        book['内容简介'] = content_lst[-1]
                if tmp == '作者简介':
                    writer_intro = item.find_next_sibling().text.strip()
                    book['作者简介'] = writer_intro
                    if '\n' in writer_intro:
                        writer_lst = writer_intro.split('\n')
                        book['作者简介'] = writer_lst[-1]
            except:
                continue
    except:
        pass

    db_book.insert_one(book)

    print(i)


for i in range(num):
    spider(i)
