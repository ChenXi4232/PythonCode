import sys
import pandas as pd


def process_xlsx_file(input_file, output_file, column_names):
    try:
        # 读取xlsx文件
        df = pd.read_excel(input_file)
        # 遍历每个指定的列名
        with open(output_file, 'w') as f:
            for column_name in column_names:
                # 检查指定的列是否存在
                if column_name not in df.columns:
                    print(f"在文件 {input_file} 中未找到列名 {column_name}")
                    continue
                # 找到该列某行为空值的索引
                null_indexes = df[df[column_name].isnull()].index
                # 选择学号和姓名列，并且筛选出空值行
                result_df = df.loc[null_indexes, ['学号']]
                # 输出到txt文件
                f.write(f"{column_name}\n")
                for index, row in result_df.iterrows():
                    f.write(f"{row['学号']}\n")
                f.write('\n')
        print(f"已将文件 {input_file} 中空值行的学号和姓名输出到 {output_file}")
    except Exception as e:
        print(f"处理文件 {input_file} 时出现错误: {str(e)}")


if __name__ == "__main__":
    # 检查参数数量是否正确
    if len(sys.argv) < 4:
        print("用法：python script.py <输入文件名> <输出文件名> <列名1> [<列名2> ...]")
        sys.exit(1)
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    column_names = sys.argv[3:]
    process_xlsx_file(input_file, output_file, column_names)
