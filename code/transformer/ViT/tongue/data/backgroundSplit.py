import cv2
import numpy as np
import os

# # 读取图像
# image = cv2.imread(
#     './dataset/an_hong/tongue_front_300110899004_2023-11-08-15-36-06.jpg', cv2.IMREAD_GRAYSCALE)

# # 使用 Canny 边缘检测算法
# edges = cv2.Canny(image, 50, 150)  # 参数可以调整以获得更好的结果

# # 设置显示窗口的大小
# window_name = 'Original Image'
# cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # 允许调整窗口大小
# cv2.resizeWindow(window_name, 600, 400)  # 设置窗口大小

# window_name = 'Detected Edges'
# cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # 允许调整窗口大小
# cv2.resizeWindow(window_name, 600, 400)  # 设置窗口大小

# # 显示原始图像和检测到的边缘
# cv2.imshow('Original Image', image)
# cv2.imshow('Detected Edges', edges)

# # 按任意键退出
# cv2.waitKey(0)
# cv2.destroyAllWindows()


def process_images_in_folder(path):
    if os.path.isdir(path) and os.path.isdir(path):
        # 遍历文件夹中的所有文件和子文件夹
        for root, dirs, files in os.walk(path):
            for file_name in files:
                # 只处理图像文件
                if file_name.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    # 构建图像文件的完整路径
                    image_path = os.path.join(root, file_name)
                    # 调用图像处理函数
                    process_image(image_path)
    elif os.path.isfile(path) and os.path.isfile(path):
        process_image(path)
    else:
        print(f"Invalid input path: {path}")


def process_image(image_path):
    # 读取图像
    image = cv2.imread(image_path)

    # 将图像转换为灰度
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 通过阈值处理将黑色部分置为白色（255）
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # 寻找轮廓
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 找到最大的轮廓
    max_contour = max(contours, key=cv2.contourArea)

    # 计算最小包围矩形
    x, y, w, h = cv2.boundingRect(max_contour)

    # 在原图上绘制矩形
    result = cv2.rectangle(image.copy(), (x, y),
                           (x + w, y + h), (0, 255, 0), 2)

    # 截取矩形框内的区域
    cropped = image[y:y+h, x:x+w]

    # 保存截取的区域
    cv2.imwrite(image_path, cropped)

    # window_name = 'Bounding Rectangle'
    # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # 允许调整窗口大小
    # cv2.resizeWindow(window_name, 300, 400)  # 设置窗口大小

    # 显示结果
    # cv2.imshow('Bounding Rectangle', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print(f"Processed image saved to {image_path}")


# 处理文件夹中的所有图像
input_folder = r'D:\PythonCode\code\transformer\ViT\tongue\test\seg_dataset\test2.jpg'

process_images_in_folder(input_folder)
