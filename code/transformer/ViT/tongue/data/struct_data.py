import os
import csv
import shutil
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
        classes = file.readline().strip().split()[1:]
        # next(file)  # Skip header
        for line in file:
            parts = line.strip().split()
            index = parts[0]
            labels = {key: chinese_to_pinyin(value)
                      for key, value in zip(classes, parts[1:])}
            data[index] = labels
    return data


def create_dataset(csv_file, txt_data, image_dir, output_dir):
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        images = os.listdir(image_dir)
        for row in reader:
            for image in images:
                image_id = int(image.split('_')[2])
                if int(row['id']) == image_id:
                    for key, value in row.items():
                        if key != 'id':
                            if len(value) > 1:
                                value = eval(value)
                            else:
                                value = [value]
                            for v in value:
                                temp = txt_data[str(v)][key]
                                if temp != 'Nan' and temp != 'zheng_chang':
                                    if not os.path.exists(os.path.join(output_dir, temp)):
                                        os.makedirs(
                                            os.path.join(output_dir, temp))
                                    shutil.copy(os.path.join(image_dir, image),
                                                os.path.join(output_dir, temp, image))
                    break

    # os.makedirs(output_dir, exist_ok=True)
    # for index, labels in txt_data.items():
    #     for key, label in labels.items():
    #         label_dir = os.path.join(output_dir, label)
    #         if label != 'Nan' and label != 'zheng_chang':
    #             os.makedirs(label_dir, exist_ok=True)

    #             with open(csv_file, 'r') as file:
    #                 reader = csv.DictReader(file)
    #                 for row in reader:
    #                     if row['id'].startswith('tongue_front_' + index):
    #                         image_file = os.path.join(
    #                             image_dir, row['id'] + '.jpg')
    #                         if os.path.exists(image_file):
    #                             os.rename(image_file, os.path.join(
    #                                 label_dir, row['id'] + '.jpg'))


def main():
    txt_file = './label.txt'
    csv_file = './multi_label_tongue.csv'
    image_dir = './seg_dataset'
    output_dir = 'dataset'

    txt_data = read_txt_file(txt_file)
    create_dataset(csv_file, txt_data, image_dir, output_dir)


if __name__ == "__main__":
    main()
