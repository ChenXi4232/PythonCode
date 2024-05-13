import os


def remove_nth_number_in_txt_files(folder_path, n):
    """
    删除指定文件夹下txt文件第一行指定的第n个数
    Args:
        folder_path (str): 文件夹路径
        n (int): 要删除的数字的位置（从1开始）

    Returns:
        None
    """
    # 检查文件夹路径是否存在
    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return

    # 遍历文件夹中的txt文件
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            # 打开文件并读取内容
            with open(file_path, 'r') as file:
                lines = file.readlines()
            # 修改第一行的内容
            if lines:
                numbers = lines[0].strip().split()
                if 1 <= n <= len(numbers):
                    del numbers[n - 1]
                    # 将修改后的内容写回文件
                    with open(file_path, 'w') as file:
                        file.write(' '.join(numbers))
                        file.write('\n')
                    print(f"Deleted the {n}th number in file '{file_name}'.")
                else:
                    print(f"Error: No {n}th number in file '{file_name}'.")
            else:
                print(f"Warning: File '{file_name}' is empty.")


# 示例用法
folder_path = "./dataset_new/tongue_shape/labels"
n = 4
remove_nth_number_in_txt_files(folder_path, n)
