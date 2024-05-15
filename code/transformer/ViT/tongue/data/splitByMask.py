import os
import cv2
import numpy as np


def segment_images_with_gray(gray_input, original_input, output_dir):

    if os.path.isdir(gray_input) and os.path.isdir(original_input):
        gray_images = os.listdir(gray_input)

        for gray_image_name in gray_images:
            # 获取伪彩色标注图像的完整路径
            gray_image_path = os.path.join(
                gray_input, gray_image_name)

            # 获取对应的原始图像名称
            original_image_name = os.path.splitext(
                gray_image_name)[0] + ".jpg"

            # 获取对应的原始图像完整路径
            original_image_path = os.path.join(original_input, original_image_name)

            # 检查原始图像是否存在
            if not os.path.exists(original_image_path):
                print(
                    f"Warning: Original image {original_image_name} not found for {gray_image_name}")
                continue

            # 读取灰度图
            gray_image = cv2.imread(gray_image_path)

            # 读取原始图像
            original_image = cv2.imread(original_image_path)

            labels = gray_image[:, :, 2]

            # 创建掩码，标签值为0的部分为True，其余部分为False
            mask = labels != 0

            segmented_image = original_image.copy()
            segmented_image[mask] = [0, 0, 0]

            # 保存分割结果图像
            output_segmented_path = os.path.join(output_dir, original_image_name)
            cv2.imwrite(output_segmented_path, segmented_image)

            print(
                f"Segmented image {original_image_name} saved to {output_segmented_path}")

    elif os.path.isfile(gray_input) and os.path.isfile(original_input):
        gray_image = cv2.imread(gray_input)
        original_image = cv2.imread(original_input)

        labels = gray_image[:, :, 2]

        mask = labels != 0

        segmented_image = original_image.copy()
        segmented_image[mask] = [0, 0, 0]

        output_segmented_path = os.path.join(output_dir, os.path.basename(original_input))
        cv2.imwrite(output_segmented_path, segmented_image)

        print(
            f"Segmented image {os.path.basename(original_input)} saved to {output_segmented_path}")

    else:
        print("Error: Invalid input path")
        return


# 原图像目录路径
original_folder = r'D:\PythonCode\code\transformer\ViT\tongue\data\test\label_data\test1.jpg'
# 蒙版目录路径
mask_folder = r'D:\PythonCode\code\transformer\ViT\tongue\mask3.jpg'
# 分割后图像保存目录路径
output_folder = '../test/seg_dataset'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

segment_images_with_gray(mask_folder, original_folder, output_folder)
