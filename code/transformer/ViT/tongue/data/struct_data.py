import os
import csv
from pypinyin import pinyin, Style


def chinese_to_pinyin(text):
    pinyin_list = pinyin(text, style=Style.NORMAL)
    result = ''
    for char, py in zip(text, pinyin_list):
        if char.isalpha():
            result += py[0] + '_'
        else:
            result += char + '_'
    return result.rstrip('_')


def read_txt_file(txt_file):
    data = {}
    with open(txt_file, 'r', encoding='utf-8') as file:
        next(file)  # Skip header
        for line in file:
            parts = line.strip().split()
            index = parts[0]
            labels = [chinese_to_pinyin(
                x) for x in parts[1:] if x != 'Nan' and x != '正常']
            data[index] = labels
    return data


def create_dataset(csv_file, txt_data, image_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for index, labels in txt_data.items():
        for label in labels:
            label_dir = os.path.join(output_dir, label)
            os.makedirs(label_dir, exist_ok=True)

            with open(csv_file, 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if row['id'].startswith('tongue_front_' + index):
                        image_file = os.path.join(
                            image_dir, row['id'] + '.jpg')
                        if os.path.exists(image_file):
                            os.rename(image_file, os.path.join(
                                label_dir, row['id'] + '.jpg'))


def main():
    txt_file = './label.txt'
    csv_file = './multi_label_tongue.csv'
    image_dir = './tongue_front'
    output_dir = 'dataset'

    txt_data = read_txt_file(txt_file)
    create_dataset(csv_file, txt_data, image_dir, output_dir)


if __name__ == "__main__":
    main()
