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
