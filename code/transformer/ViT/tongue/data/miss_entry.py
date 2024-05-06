import os
import pandas as pd


text = 'tongue_morphology'
# 读取 CSV 文件中的 id 列
df = pd.read_csv('multi_label_tongue.csv')
id_list = df['id'].tolist()
entry_list = df[text].tolist()

# 设置文件夹路径
folder_path = './dataset_new/tongue_color/'
output_file = 'missing_entry.txt'

# 扫描文件夹中所有以 'tongue_front' 开头的 jpg 文件
missing_entries = []
missing_entries_id = id_list
for root, dirs, files in os.walk(folder_path):
    for filename in files:
        if filename.startswith('tongue_front') and filename.endswith('.jpg'):
            # 提取文件名中的 id
            parts = filename.split('_')
            file_id = int(parts[2])

            # 检查 id 是否在 CSV 文件中
            if file_id in id_list:
                missing_entries_id.remove(file_id)

temp = missing_entries_id.copy()

# for id in temp:
#     temp1 = df.loc[df['id'] == id, text].values[0]
#     length = len(df.loc[df['id'] == id, text].values[0]) if type(df.loc[df['id'] == id, text].values[0]) == str else 1
#     if length == 1 and int(temp1) == 1:
#         missing_entries_id.remove(id)

# 将不在 CSV 文件中的文件名写入 'missing_entry.txt' 文件中
with open(output_file, 'w') as file:
    # for entry in missing_entries:
    #     file.write(entry + '\n')
    for id in missing_entries_id:
        file.write(str(id) + '\n')

print("Missing entries have been written to 'missing_entry.txt'.")
