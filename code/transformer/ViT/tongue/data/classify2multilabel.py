import os
import shutil


def organize_dataset(input_folder, output_folder):
    # 创建输出文件夹
    images_folder = os.path.join(output_folder, 'images')
    labels_folder = os.path.join(output_folder, 'labels')
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(labels_folder, exist_ok=True)

    # 获取所有类别文件夹的名称
    classes = sorted(os.listdir(input_folder))

    with open(os.path.join(output_folder, 'classes.txt'), 'w') as f:
        f.write(' '.join(classes))

    images_labels = {}
    images_set = set()
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                if file in images_set:
                    continue
                dest_path = os.path.join(images_folder, file)
                shutil.copy(os.path.join(root, file), dest_path)
                images_set.add(os.path.splitext(file)[0])

    for image in images_set:
        images_labels[image] = [0] * len(classes)

    # 遍历输入文件夹中的每个类别文件夹
    for idx, class_folder in enumerate(classes):
        for filename in os.listdir(os.path.join(input_folder, class_folder)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                images_labels[os.path.splitext(filename)[0]][idx] = 1

    for image, labels in images_labels.items():
        with open(os.path.join(labels_folder, image + '.txt'), 'w') as f:
            f.write(' '.join(map(str, labels)))


# 设置输入和输出文件夹路径
input_folder = './dataset/tongue_morphology'
output_folder = './dataset_new/tongue_morphology'

# 整理数据集
organize_dataset(input_folder, output_folder)
