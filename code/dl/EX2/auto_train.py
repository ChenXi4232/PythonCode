import os
import re


def remove_comments(file_path, start_line, end_line):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for i in range(start_line - 1, end_line):
        index = lines[i].find('#')  # 找到第一个 # 的索引位置
        lines[i] = lines[i][:index] + lines[i][index + 2:]  # 删除注释符号及其之前的内容

    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(lines)


def add_comments(file_path, start_line, end_line):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for i in range(start_line - 1, end_line):
        first_non_space_index = len(lines[i]) - len(lines[i].lstrip())
        # 在第一个非空格字符后添加注释符号，并保留原有的空格
        lines[i] = lines[i][:first_non_space_index] + \
            '# ' + lines[i][first_non_space_index:]

    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(lines)


def modify_line(file_path, line_number, new_content):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    if line_number <= len(lines):
        lines[line_number - 1] = new_content + '\n'  # 注意行号是从1开始计数的
    else:
        print("Error: Line number exceeds total number of lines in the file.")

    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(lines)


def replace_in_file(file_path, pattern, replacement):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    with open(file_path, 'w', encoding='utf-8') as file:
        for line in lines:
            modified_line = re.sub(pattern, replacement, line)
            file.write(modified_line)


def execute_git_commands_in_directory(directory, git_commands):
    # 获取当前工作目录
    original_directory = os.getcwd()
    print(original_directory)

    try:
        # 切换到指定目录
        os.chdir(directory)
        print(os.getcwd())

        # 依次执行 git 命令
        for git_command in git_commands:
            os.system(git_command)

    finally:
        # 切换回原来的工作目录
        os.chdir(original_directory)
        print(os.getcwd())


directory = 'D:/PythonCode'  # 替换为你想要执行 git 命令的目录路径
git_commands = [
    'git add .',
    'git commit -m "update"',
    'git push'
]

execute_git_commands_in_directory(directory, git_commands)

# '''
# depth2-1_kernel3-1_dropout1-0.6_normData-bn_lrCLR0.1-0.6-up0.25_Aug
# '''

# os.system('python CNN2.py')
# execute_git_commands_in_directory(directory, git_commands)

# '''
# depth3-2_kernel1-0_dropout1-0.6_normData-bn_lrCLR0.1-0.6-up0.25_Aug
# '''

# remove_comments('CNN2.py', 299, 301)
# modify_line('CNN2.py', 292, '        self.linear = nn.Linear(512, 10)')
# replace_in_file('CNN2.py', 'kernel_size=3', 'kernel_size=1')
# replace_in_file('CNN2.py', 'padding=1', 'padding=0')
# modify_line('CNN2.py', 164,
#             '    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),')
# modify_line('CNN2.py', 476,
#             'file_name_prefix = \'depth3-2_kernel1-0_dropout1-0.6_normData-bn_lrCLR0.1-0.6-up0.25_Aug\'')
# os.system('python CNN2.py')
# execute_git_commands_in_directory(directory, git_commands)

# '''
# depth3-2_kernel3-1_dropout1-0.6_normData-bn_lrC0.1_Aug
# '''

# replace_in_file('CNN2.py', 'batch_size=768', 'batch_size=128')
# replace_in_file('CNN2.py', 'kernel_size=1', 'kernel_size=3')
# replace_in_file('CNN2.py', 'padding=0', 'padding=1')
# add_comments('CNN2.py', 38, 46)
# remove_comments('CNN2.py', 49, 56)
# add_comments('CNN2.py', 59, 66)
# remove_comments('CNN2.py', 69, 75)
# add_comments('CNN2.py', 371, 373)
# add_comments('CNN2.py', 376, 376)
# remove_comments('CNN2.py', 377, 377)
# add_comments('CNN2.py', 434, 434)
# remove_comments('CNN2.py', 435, 435)
# modify_line('CNN2.py', 476,
#             'file_name_prefix = \'depth3-2_kernel3-1_dropout1-0.6_normData-bn_lrC0.1_Aug\'')
# os.system('python CNN2.py')
# execute_git_commands_in_directory(directory, git_commands)

# '''
# depth3-2_kernel3-1_dropout1-0.6_normData-bn_lrCLR0.1-0.6-up0.25_Aug
# '''

# replace_in_file('CNN2.py', 'batch_size=128', 'batch_size=768')
# remove_comments('CNN2.py', 38, 46)
# add_comments('CNN2.py', 49, 56)
# remove_comments('CNN2.py', 59, 66)
# add_comments('CNN2.py', 69, 75)
# remove_comments('CNN2.py', 371, 373)
# remove_comments('CNN2.py', 376, 376)
# add_comments('CNN2.py', 377, 377)
# remove_comments('CNN2.py', 434, 434)
# add_comments('CNN2.py', 435, 435)
# modify_line('CNN2.py', 476,
#             'file_name_prefix = \'depth3-2_kernel3-1_dropout1-0.6_normData-bn_lrCLR0.1-0.6-up0.25_Aug\'')
# os.system('python CNN2.py')
# execute_git_commands_in_directory(directory, git_commands)

# '''
# depth3-2_kernel3-1_dropout1-0.6_normData-bn_lrStep0.2-0.5_Aug
# '''

# replace_in_file('CNN2.py', 'batch_size=768', 'batch_size=128')
# remove_comments('CNN2.py', 368, 369)
# add_comments('CNN2.py', 371, 373)
# modify_line('CNN2.py', 476,
#             'file_name_prefix = \'depth3-2_kernel3-1_dropout1-0.6_normData-bn_lrStep0.2-0.5_Aug\'')
# os.system('python CNN2.py')
# execute_git_commands_in_directory(directory, git_commands)

# '''
# depth3-2_kernel3-1_dropout1-0.6_normData-in_lrCLR0.1-0.6-up0.25_Aug
# '''

# replace_in_file('CNN2.py', 'batch_size=128', 'batch_size=768')
# replace_in_file('CNN2.py', 'nn.BatchNorm2d', 'nn.InstanceNorm2d')
# add_comments('CNN2.py', 368, 369)
# remove_comments('CNN2.py', 371, 373)
# modify_line('CNN2.py', 476,
#             'file_name_prefix = \'depth3-2_kernel3-1_dropout1-0.6_normData-in_lrCLR0.1-0.6-up0.25_Aug\'')
# os.system('python CNN2.py')
# execute_git_commands_in_directory(directory, git_commands)

# '''
# depth3-2_kernel3-1_dropout1-0.6_normData-Notn_lrCLR0.1-0.6-up0.25_Aug
# '''

# replace_in_file('CNN2.py', 'nn.InstanceNorm2d', 'nn.BatchNorm2d')
# add_comments('CNN2.py', 220, 220)
# add_comments('CNN2.py', 225, 225)
# add_comments('CNN2.py', 239, 239)
# add_comments('CNN2.py', 246, 246)
# add_comments('CNN2.py', 255, 255)
# add_comments('CNN2.py', 263, 263)
# modify_line('CNN2.py', 476,
#             'file_name_prefix = \'depth3-2_kernel3-1_dropout1-0.6_normData-Notn_lrCLR0.1-0.6-up0.25_Aug\'')
# os.system('python CNN2.py')
# execute_git_commands_in_directory(directory, git_commands)

# '''
# depth3-2_kernel3-1_dropout1-0.6_NotnormData-bn_lrCLR0.1-0.6-up0.25_Aug
# '''

# add_comments('CNN2.py', 168, 168)
# add_comments('CNN2.py', 174, 174)
# remove_comments('CNN2.py', 220, 220)
# remove_comments('CNN2.py', 225, 225)
# remove_comments('CNN2.py', 239, 239)
# remove_comments('CNN2.py', 246, 246)
# remove_comments('CNN2.py', 255, 255)
# remove_comments('CNN2.py', 263, 263)
# modify_line('CNN2.py', 476,
#             'file_name_prefix = \'depth3-2_kernel3-1_dropout1-0.6_NotnormData-bn_lrCLR0.1-0.6-up0.25_Aug\'')
# os.system('python CNN2.py')
# execute_git_commands_in_directory(directory, git_commands)

# '''
# depth3-2_kernel3-1_dropout0-0_normData-bn_lrCLR0.1-0.6-up0.25_Aug
# '''

# remove_comments('CNN2.py', 168, 168)
# remove_comments('CNN2.py', 174, 174)
# add_comments('CNN2.py', 304, 304)
# remove_comments('CNN2.py', 308, 308)
# remove_comments('CNN2.py', 311, 311)
# modify_line('CNN2.py', 476,
#             'file_name_prefix = \'depth3-2_kernel3-1_dropout0-0_normData-bn_lrCLR0.1-0.6-up0.25_Aug\'')
# os.system('python CNN2.py')
# execute_git_commands_in_directory(directory, git_commands)

# '''
# depth3-2_kernel3-1_dropout3-0.2_normData-bn_lrCLR0.1-0.6-up0.25_Aug
# '''

# remove_comments('CNN2.py', 305, 305)
# remove_comments('CNN2.py', 309, 309)
# remove_comments('CNN2.py', 312, 312)
# modify_line('CNN2.py', 285, '        self.dropout1 = nn.Dropout(0.2)')
# modify_line('CNN2.py', 476,
#             'file_name_prefix = \'depth3-2_kernel3-1_dropout3-0.2_normData-bn_lrCLR0.1-0.6-up0.25_Aug\'')
# os.system('python CNN2.py')
# execute_git_commands_in_directory(directory, git_commands)

# '''
# depth3-2_kernel3-1_dropout3-0.4_normData-bn_lrCLR0.1-0.6-up0.25_Aug
# '''

# modify_line('CNN2.py', 285, '        self.dropout1 = nn.Dropout(0.4)')
# modify_line('CNN2.py', 476,
#             'file_name_prefix = \'depth3-2_kernel3-1_dropout3-0.4_normData-bn_lrCLR0.1-0.6-up0.25_Aug\'')
# os.system('python CNN2.py')
# execute_git_commands_in_directory(directory, git_commands)

# '''
# depth3-2_kernel3-1_dropout3-0.6_normData-bn_lrCLR0.1-0.6-up0.25_Aug
# '''

# modify_line('CNN2.py', 285, '        self.dropout1 = nn.Dropout(0.6)')
# modify_line('CNN2.py', 476,
#             'file_name_prefix = \'depth3-2_kernel3-1_dropout3-0.6_normData-bn_lrCLR0.1-0.6-up0.25_Aug\'')
# os.system('python CNN2.py')
# execute_git_commands_in_directory(directory, git_commands)

# '''
# depth3-2_kernel3-1_dropout3-0.8_normData-bn_lrCLR0.1-0.6-up0.25_Aug
# '''

# modify_line('CNN2.py', 285, '        self.dropout1 = nn.Dropout(0.8)')
# modify_line('CNN2.py', 476,
#             'file_name_prefix = \'depth3-2_kernel3-1_dropout3-0.8_normData-bn_lrCLR0.1-0.6-up0.25_Aug\'')
# os.system('python CNN2.py')
# execute_git_commands_in_directory(directory, git_commands)

# restart

'''
depth3-2_kernel3-1_dropout3-0.2_normData-bn_lrCLR0.1-0.6-up0.25_Aug
'''

modify_line('CNN2.py', 285, '        self.dropout1 = nn.Dropout(0.2)')
modify_line('CNN2.py', 476,
            'file_name_prefix = \'depth3-2_kernel3-1_dropout3-0.2_normData-bn_lrCLR0.1-0.6-up0.25_Aug\'')
os.system('python CNN2.py')
execute_git_commands_in_directory(directory, git_commands)

'''
depth3-2_kernel3-1_dropout3-0.4_normData-bn_lrCLR0.1-0.6-up0.25_Aug
'''

modify_line('CNN2.py', 285, '        self.dropout1 = nn.Dropout(0.4)')
modify_line('CNN2.py', 476,
            'file_name_prefix = \'depth3-2_kernel3-1_dropout3-0.4_normData-bn_lrCLR0.1-0.6-up0.25_Aug\'')
os.system('python CNN2.py')
execute_git_commands_in_directory(directory, git_commands)

'''
depth3-2_kernel5-2_dropout1-0.6_normData-bn_lrCLR0.1-0.6-up0.25_Aug
'''

add_comments('CNN2.py', 308, 309)
add_comments('CNN2.py', 311, 312)
modify_line('CNN2.py', 285, '        self.dropout1 = nn.Dropout(0.6)')
replace_in_file('CNN2.py', 'kernel_size=3', 'kernel_size=5')
replace_in_file('CNN2.py', 'padding=1', 'padding=2')
modify_line('CNN2.py', 164,
            '    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),')
modify_line('CNN2.py', 476,
            'file_name_prefix = \'depth3-2_kernel5-2_dropout1-0.6_normData-bn_lrCLR0.1-0.6-up0.25_Aug\'')
# os.system('python CNN2.py')
execute_git_commands_in_directory(directory, git_commands)

'''
depth5-3_kernel3-1_dropout1-0.6_normData-bn_lrCLR0.1-0.6-up0.25_Aug
'''

replace_in_file('CNN2.py', 'kernel_size=5', 'kernel_size=3')
replace_in_file('CNN2.py', 'padding=2', 'padding=1')
remove_comments('CNN2.py', 302, 304)
modify_line('CNN2.py', 292, '        self.linear = nn.Linear(2048, 10)')
modify_line('CNN2.py', 476,
            'file_name_prefix = \'depth5-3_kernel3-1_dropout1-0.6_normData-bn_lrCLR0.1-0.6-up0.25_Aug\'')
os.system('python CNN2.py')
execute_git_commands_in_directory(directory, git_commands)

'''
depth3-2_kernel3-1_dropout1-0.6_normData-bn_lrCLR0.1-0.6-up0.25_NotAug
'''

add_comments('CNN2.py', 302, 304)
add_comments('CNN2.py', 159, 166)
add_comments('CNN2.py', 169, 169)
modify_line('CNN2.py', 292, '        self.linear = nn.Linear(512, 10)')
modify_line('CNN2.py', 476,
            'file_name_prefix = \'depth3-2_kernel3-1_dropout1-0.6_normData-bn_lrCLR0.1-0.6-up0.25_NotAug\'')
os.system('python CNN2.py')
execute_git_commands_in_directory(directory, git_commands)

'''

'''
