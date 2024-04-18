import os
import re


def remove_comments(file_path, start_line, end_line):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    for i in range(start_line - 1, end_line):
        lines[i] = lines[i].lstrip('# ')  # 去掉行首的注释符号

    with open(file_path, 'w') as file:
        file.writelines(lines)


def add_comments(file_path, start_line, end_line):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    for i in range(start_line - 1, end_line):
        lines[i] = '# ' + lines[i]  # 在行首添加注释符号

    with open(file_path, 'w') as file:
        file.writelines(lines)


def modify_line(file_path, line_number, new_content):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    if line_number <= len(lines):
        lines[line_number - 1] = new_content + '\n'  # 注意行号是从1开始计数的
    else:
        print("Error: Line number exceeds total number of lines in the file.")

    with open(file_path, 'w') as file:
        file.writelines(lines)


def replace_in_file(file_path, pattern, replacement):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    with open(file_path, 'w') as file:
        for line in lines:
            modified_line = re.sub(pattern, replacement, line)
            file.write(modified_line)


'''
depth2-1_kernel3-1_dropout1-0.6_normData-bn_lrCLR0.1-0.6-up0.25_Aug
'''

os.system('python3 CNN2.py')

'''
depth3-2_kernel1-0_dropout1-0.6_normData-bn_lrCLR0.1-0.6-up0.25_Aug
'''

remove_comments('CNN2.py', 298, 300)
modify_line('CNN2.py', 291, '        self.linear = nn.Linear(512, 10)')
replace_in_file('CNN2.py', 'kernel_size=3', 'kernel_size=1')
replace_in_file('CNN2.py', 'padding=1', 'padding=0')
modify_line('CNN2.py', 162,
            '    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),')
modify_line('CNN2.py', 475,
            'file_name_prefix = \'depth3-2_kernel1-0_dropout1-0.6_normData-bn_lrCLR0.1-0.6-up0.25_Aug\'')
os.system('python3 CNN2.py')

'''
depth3-2_kernel3-1_dropout1-0.6_normData-bn_lrC0.1_Aug
'''

replace_in_file('CNN2.py', 'kernel_size=1', 'kernel_size=3')
replace_in_file('CNN2.py', 'padding=0', 'padding=1')
add_comments('CNN2.py', 37, 45)
remove_comments('CNN2.py', 48, 55)
add_comments('CNN2.py', 58, 65)
remove_comments('CNN2.py', 68, 74)
add_comments('CNN2.py', 370, 372)
add_comments('CNN2.py', 375, 375)
remove_comments('CNN2.py', 376, 376)
add_comments('CNN2.py', 433, 433)
remove_comments('CNN2.py', 434, 434)
modify_line('CNN2.py', 475,
            'file_name_prefix = \'depth3-2_kernel3-1_dropout1-0.6_normData-bn_lrC0.1_Aug\'')
os.system('python3 CNN2.py')

'''
depth3-2_kernel3-1_dropout1-0.6_normData-bn_lrCLR0.1-0.6-up0.25_Aug
'''

remove_comments('CNN2.py', 37, 45)
add_comments('CNN2.py', 48, 55)
remove_comments('CNN2.py', 58, 65)
add_comments('CNN2.py', 68, 74)
remove_comments('CNN2.py', 370, 372)
remove_comments('CNN2.py', 375, 375)
add_comments('CNN2.py', 376, 376)
remove_comments('CNN2.py', 433, 433)
add_comments('CNN2.py', 434, 434)
modify_line('CNN2.py', 475,
            'file_name_prefix = \'depth3-2_kernel3-1_dropout1-0.6_normData-bn_lrCLR0.1-0.6-up0.25_Aug\'')
os.system('python3 CNN2.py')

'''
depth3-2_kernel3-1_dropout1-0.6_normData-bn_lrStep0.2-0.5_Aug
'''

remove_comments('CNN2.py', 367, 368)
add_comments('CNN2.py', 370, 372)
modify_line('CNN2.py', 475,
            'file_name_prefix = \'depth3-2_kernel3-1_dropout1-0.6_normData-bn_lrStep0.2-0.5_Aug\'')
os.system('python3 CNN2.py')

'''
depth3-2_kernel3-1_dropout1-0.6_normData-in_lrCLR0.1-0.6-up0.25_Aug
'''

replace_in_file('CNN2.py', 'nn.BatchNorm2d', 'nn.InstanceNorm2d')
add_comments('CNN2.py', 367, 368)
remove_comments('CNN2.py', 370, 372)
modify_line('CNN2.py', 475,
            'file_name_prefix = \'depth3-2_kernel3-1_dropout1-0.6_normData-in_lrCLR0.1-0.6-up0.25_Aug\'')
os.system('python3 CNN2.py')

'''
depth3-2_kernel3-1_dropout1-0.6_normData-Notn_lrCLR0.1-0.6-up0.25_Aug
'''

replace_in_file('CNN2.py', 'nn.InstanceNorm2d', 'nn.BatchNorm2d')
add_comments('CNN2.py', 219, 219)
add_comments('CNN2.py', 224, 224)
add_comments('CNN2.py', 238, 238)
add_comments('CNN2.py', 245, 245)
add_comments('CNN2.py', 254, 254)
add_comments('CNN2.py', 262, 262)
modify_line('CNN2.py', 475,
            'file_name_prefix = \'depth3-2_kernel3-1_dropout1-0.6_normData-Notn_lrCLR0.1-0.6-up0.25_Aug\'')
os.system('python3 CNN2.py')

'''
depth3-2_kernel3-1_dropout1-0.6_NotnormData-bn_lrCLR0.1-0.6-up0.25_Aug
'''

add_comments('CNN2.py', 167, 167)
add_comments('CNN2.py', 173, 173)
remove_comments('CNN2.py', 219, 219)
remove_comments('CNN2.py', 224, 224)
remove_comments('CNN2.py', 238, 238)
remove_comments('CNN2.py', 245, 245)
remove_comments('CNN2.py', 254, 254)
remove_comments('CNN2.py', 262, 262)
modify_line('CNN2.py', 475,
            'file_name_prefix = \'depth3-2_kernel3-1_dropout1-0.6_NotnormData-bn_lrCLR0.1-0.6-up0.25_Aug\'')
os.system('python3 CNN2.py')

'''
depth3-2_kernel3-1_dropout0-0_normData-bn_lrCLR0.1-0.6-up0.25_Aug
'''

remove_comments('CNN2.py', 167, 167)
remove_comments('CNN2.py', 173, 173)
add_comments('CNN2.py', 304, 304)
remove_comments('CNN2.py', 307, 307)
remove_comments('CNN2.py', 310, 310)
modify_line('CNN2.py', 475,
            'file_name_prefix = \'depth3-2_kernel3-1_dropout0-0_normData-bn_lrCLR0.1-0.6-up0.25_Aug\'')
os.system('python3 CNN2.py')

'''
depth3-2_kernel3-1_dropout3-0.2_normData-bn_lrCLR0.1-0.6-up0.25_Aug
'''

remove_comments('CNN2.py', 304, 304)
remove_comments('CNN2.py', 308, 308)
remove_comments('CNN2.py', 311, 311)
modify_line('CNN2.py', 284, '        self.dropout1 = nn.Dropout(0.2)')
modify_line('CNN2.py', 475,
            'file_name_prefix = \'depth3-2_kernel3-1_dropout3-0.2_normData-bn_lrCLR0.1-0.6-up0.25_Aug\'')
os.system('python3 CNN2.py')

'''
depth3-2_kernel3-1_dropout3-0.4_normData-bn_lrCLR0.1-0.6-up0.25_Aug
'''

modify_line('CNN2.py', 284, '        self.dropout1 = nn.Dropout(0.4)')
modify_line('CNN2.py', 475,
            'file_name_prefix = \'depth3-2_kernel3-1_dropout3-0.4_normData-bn_lrCLR0.1-0.6-up0.25_Aug\'')
os.system('python3 CNN2.py')

'''
depth3-2_kernel3-1_dropout3-0.6_normData-bn_lrCLR0.1-0.6-up0.25_Aug
'''

modify_line('CNN2.py', 284, '        self.dropout1 = nn.Dropout(0.6)')
modify_line('CNN2.py', 475,
            'file_name_prefix = \'depth3-2_kernel3-1_dropout3-0.6_normData-bn_lrCLR0.1-0.6-up0.25_Aug\'')
os.system('python3 CNN2.py')

'''
depth3-2_kernel3-1_dropout3-0.8_normData-bn_lrCLR0.1-0.6-up0.25_Aug
'''

modify_line('CNN2.py', 284, '        self.dropout1 = nn.Dropout(0.8)')
modify_line('CNN2.py', 475,
            'file_name_prefix = \'depth3-2_kernel3-1_dropout3-0.8_normData-bn_lrCLR0.1-0.6-up0.25_Aug\'')
os.system('python3 CNN2.py')

'''
depth3-2_kernel5-2_dropout1-0.6_normData-bn_lrCLR0.1-0.6-up0.25_Aug
'''

add_comments('CNN2.py', 307, 308)
add_comments('CNN2.py', 310, 311)
modify_line('CNN2.py', 284, '        self.dropout1 = nn.Dropout(0.6)')
replace_in_file('CNN2.py', 'kernel_size=3', 'kernel_size=5')
replace_in_file('CNN2.py', 'padding=1', 'padding=2')
modify_line('CNN2.py', 162,
            '    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),')
modify_line('CNN2.py', 475,
            'file_name_prefix = \'depth3-2_kernel5-2_dropout1-0.6_normData-bn_lrCLR0.1-0.6-up0.25_Aug\'')
os.system('python3 CNN2.py')

'''
depth5-3_kernel3-1_dropout1-0.6_normData-bn_lrCLR0.1-0.6-up0.25_Aug
'''

replace_in_file('CNN2.py', 'kernel_size=5', 'kernel_size=3')
replace_in_file('CNN2.py', 'padding=2', 'padding=1')
remove_comments('CNN2.py', 301, 303)
modify_line('CNN2.py', 475,
            'file_name_prefix = \'depth5-3_kernel3-1_dropout1-0.6_normData-bn_lrCLR0.1-0.6-up0.25_Aug\'')
os.system('python3 CNN2.py')

'''
depth3-2_kernel3-1_dropout1-0.6_normData-bn_lrCLR0.1-0.6-up0.25_NotAug
'''

add_comments('CNN2.py', 301, 303)
add_comments('CNN2.py', 158, 165)
add_comments('CNN2.py', 168, 168)
modify_line('CNN2.py', 475,
            'file_name_prefix = \'depth3-2_kernel3-1_dropout1-0.6_normData-bn_lrCLR0.1-0.6-up0.25_NotAug\'')
os.system('python3 CNN2.py')
