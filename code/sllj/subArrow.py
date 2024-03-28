import argparse

# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(
    description='Replace "->" with "&rightarrow;" in input file and save to output file')

# 添加命令行参数
parser.add_argument('input', help='Input file name')
parser.add_argument('output', help='Output file name')

# 解析命令行参数
args = parser.parse_args()

# 读取输入文件内容，替换字符串，并写入输出文件
with open(args.input, 'r', encoding='utf-8') as input_file:
    content = input_file.read()
    replaced_content = content.replace('->', ' &rightarrow; ')
    with open(args.output, 'w', encoding='utf-8') as output_file:
        output_file.write(replaced_content)

print(
    f'Successfully replaced "->" with "&rightarrow;" and saved to {args.output}')
