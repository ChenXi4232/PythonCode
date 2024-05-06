import cv2
import numpy as np

# 读取图片
image = cv2.imread(
    "./annotations/tongue_front_0000351212_2023-10-21-11-21-13.png")

# 将图片转换为一维数组
pixels = image.reshape(-1, 3)

# 获取唯一的像素值
unique_pixels = np.unique(pixels, axis=0)

# 打印像素值
print("Unique pixel values:")
for pixel in unique_pixels:
    print(pixel)
