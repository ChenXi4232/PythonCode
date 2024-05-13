import cv2
import numpy as np

import os
import cv2
import numpy as np


def post_process_tongue_segmentation(input_path, output_path=None):
    # 如果输入路径是文件夹，则处理文件夹中的所有图像
    if os.path.isdir(input_path):
        files = os.listdir(input_path)
        for file in files:
            image_path = os.path.join(input_path, file)
            if os.path.isfile(image_path):
                post_process_single_image(image_path, output_path)
    # 如果输入路径是文件名，则处理单张图像
    elif os.path.isfile(input_path):
        post_process_single_image(input_path, output_path)
    else:
        print("Invalid input path.")


def post_process_single_image(input_image_path, output_path=None):
    # 读取舌头分割的二值图像
    segmented_tongue = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

    # 边缘平滑
    smoothed_tongue = cv2.GaussianBlur(segmented_tongue, (5, 5), 0)

    # 形态学膨胀操作填充凹陷
    kernel = np.ones((5, 5), np.uint8)
    dilated_tongue = cv2.dilate(smoothed_tongue, kernel, iterations=15)

    # 形态学腐蚀操作填充凸起
    eroded_tongue = cv2.erode(dilated_tongue, kernel, iterations=20)

    # 如果指定了输出路径，则保存处理后的图像
    if output_path:
        output_image_path = output_path if os.path.isdir(
            output_path) else os.path.dirname(output_path)
        output_image_name = os.path.basename(input_image_path)
        output_image_path = os.path.join(output_image_path, output_image_name)
        cv2.imwrite(output_image_path, eroded_tongue)
        print(f"Processed image saved to {output_image_path}")


# 示例用法
# 对单张图像进行后处理
post_process_tongue_segmentation(
    r'D:\PythonCode\code\transformer\ViT\tongue\mask.jpg', output_path=r'D:\PythonCode\code\transformer\ViT\tongue')
# # 对文件夹中的所有图像进行后处理，并保存到指定文件夹
# post_process_tongue_segmentation(
#     'tongue_segmentation_masks_folder', output_path='processed_tongue_segmentation_masks_folder')
