import os
import cv2
import numpy as np


def segment_images_with_gray(gray_dir, images_dir, output_dir):

    gray_images = os.listdir(gray_dir)

    for gray_image_name in gray_images:
        # 获取伪彩色标注图像的完整路径
        gray_image_path = os.path.join(
            gray_dir, gray_image_name)

        # 获取对应的原始图像名称
        original_image_name = os.path.splitext(
            gray_image_name)[0] + ".jpg"

        # 获取对应的原始图像完整路径
        original_image_path = os.path.join(images_dir, original_image_name)

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


# 原图像目录路径
original_folder = './tongue_front'
# 蒙版目录路径
mask_folder = './annotations'
# 分割后图像保存目录路径
output_folder = './seg_dataset'

segment_images_with_gray(mask_folder, original_folder, output_folder)
