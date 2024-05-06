import os
import shutil


def find_missing_images(src_folder, reference_folder, dest_folder):
    # 获取 reference_folder 及其子文件夹下的所有图片路径
    reference_images = []
    for root, dirs, files in os.walk(reference_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                if file not in reference_images:
                    reference_images.append(file)

    # 获取 src_folder 及其子文件夹下的所有图片路径
    src_images = []
    for root, dirs, files in os.walk(src_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                src_images.append(os.path.join(root, file))

    # 检查 src_images 中哪些图片不在 reference_images 中
    missing_images = []
    for src_image in src_images:
        if os.path.basename(src_image) not in reference_images:
            missing_images.append(src_image)

    # 将 missing_images 复制到 dest_folder 中
    for missing_image in missing_images:
        # 构建目标文件夹中的子文件夹路径
        # relative_path = os.path.relpath(missing_image, src_folder)
        # dest_subfolder = os.path.join(
        #     dest_folder, os.path.dirname(relative_path))
        # 确保目标文件夹中的子文件夹存在
        os.makedirs(dest_folder, exist_ok=True)
        # 复制文件
        shutil.copy(missing_image, dest_folder)

    return missing_images


# 指定源文件夹、参考文件夹和目标文件夹路径
src_folder = './seg_dataset'
reference_folder = './dataset/coating_state'
dest_folder = './dataset/coating_state/zheng_chang'

# 查找并复制不在参考文件夹中的图片
missing_images = find_missing_images(src_folder, reference_folder, dest_folder)

print("以下图片不在参考文件夹中，已复制到目标文件夹中：")
for image in missing_images:
    print(image)
