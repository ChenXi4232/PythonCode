import os
import random
from PIL import Image


def read_color_values_from_file(file_path):
    """
    从文件中读取颜色数值
    Args:
        file_path (str): 文件路径

    Returns:
        dict: 包含类别及其对应颜色数值的字典
    """
    color_values = {}
    current_category = None
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:  # 跳过空行
                if line.isalpha():
                    current_category = line
                    color_values[current_category] = []
                else:
                    rgb_values = [int(value) for value in line.split(',')]
                    color_values[current_category].append(rgb_values)
    return color_values


def generate_color_blocks(color_values, output_dir, block_size, total_blocks_per_category):
    """
    生成色块图像并保存到文件
    Args:
        color_values (dict): 包含类别及其对应颜色数值的字典
        output_dir (str): 输出目录
        block_size (int): 色块大小
        total_blocks_per_category (int): 每个类别生成的总色块数量
    """
    for category, values in color_values.items():
        category_dir = os.path.join(output_dir, category)
        os.makedirs(category_dir, exist_ok=True)

        num_values = len(values)
        num_blocks_generated = 0

        while num_blocks_generated < total_blocks_per_category:
            for rgb_values in values:
                if num_blocks_generated >= total_blocks_per_category:
                    break

                color = tuple(rgb_values)
                image = Image.new("RGB", (block_size, block_size), color)
                file_name = f"{category}_color_{num_blocks_generated}.png"
                file_path = os.path.join(category_dir, file_name)
                image.save(file_path)
                num_blocks_generated += 1


def main():
    file_path = "./save_coating_color_paras.txt"
    output_dir = "./data/color_dataset/coating_color"
    block_size = 224
    total_blocks_per_category = 3000  # 每个类别生成的总色块数量

    # 从文件中读取颜色数值
    color_values = read_color_values_from_file(file_path)

    # 生成色块图像并保存到文件
    generate_color_blocks(color_values, output_dir,
                          block_size, total_blocks_per_category)


if __name__ == "__main__":
    main()
