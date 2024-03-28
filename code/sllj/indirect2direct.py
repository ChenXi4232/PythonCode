import re
import regex


def find_and_remove_substring(input_string, substring):
    index = input_string.find(substring)
    if index != -1:
        return input_string[:index] + input_string[index+len(substring):], index
    else:
        return "Substring not found in the input string."


def is_arrow_concat(expression):
    """检查表达式是否为箭头连接的两个表达式"""
    pattern = r'\((?>[^()]|(?R))*\)'
    matchs = regex.findall(pattern, expression)
    if len(matchs) == 2:
        return True
    elif len(matchs) == 1:
        if '->' in find_and_remove_substring(expression, matchs[0])[0]:
            return True
    else:
        if '->' in expression:
            return True
    return False


def is_need_pth(expression):
    """检查表达式是否需要加括号"""
    pattern = r'~+\(.*\)'
    if '->' not in expression:
        return False
    elif re.match(pattern, expression):
        return False
    else:
        return True


def extract_expression_by_arrow(expression):
    """根据箭头提取表达式"""
    pattern1 = r'\(((?>[^()]|(?R))*)\)'
    matchs1 = regex.findall(pattern1, expression)
    if matchs1:
        len1 = len(matchs1)
    else:
        len1 = 0
    pattern2 = r'(~+\((?>[^()]|(?R))*\))'
    matchs2 = regex.findall(pattern2, expression)
    if matchs2:
        len2 = len(matchs2)
    else:
        len2 = 0
    if len1 == 0 and len2 == 0:
        return expression.split('->')
    elif len1 == 1 and len2 == 0:
        temp_expr1, temp_index1 = find_and_remove_substring(
            expression, matchs1[0])
        temp_expr2, temp_index2 = find_and_remove_substring(
            temp_expr1, '->')
        output_expr = regex.findall(r'[^()]+', temp_expr2)[0]
        output = (output_expr, matchs1[0]) if temp_index2 < temp_index1 else (
            matchs1[0], output_expr)
        return output
    elif len1 == 0 and len2 == 1:
        temp_expr1, temp_index1 = find_and_remove_substring(
            expression, matchs2[0])
        temp_expr2, temp_index2 = find_and_remove_substring(
            temp_expr1, '->')
        output_expr = regex.findall(r'[^()]+', temp_expr2)[0]
        output = (matchs2[0], output_expr) if temp_index1 < temp_index2 else (
            output_expr, matchs2[0])
        return output
    elif len1 == 1 and len2 == 1:
        temp_expr1, temp_index1 = find_and_remove_substring(
            expression, matchs1[0])
        temp_expr2, temp_index2 = find_and_remove_substring(
            temp_expr1, matchs2[0])
        output = (matchs1[0], matchs2[0]) if temp_index1 < temp_index2 else (
            matchs2[0], matchs1[0])
        return output
    elif len1 == 2:
        return matchs1
    elif len2 == 2:
        return matchs2
    else:
        return None


def replace_string_with_regex(input_string):
    output_regex = re.sub(r'\(', r'\\(', input_string)
    output_regex = re.sub(r'\)', r'\\)', output_regex)
    output_regex += r'$'

    replaced_chars = {}
    chars = re.findall(r'[A-Za-z]', input_string)

    for char in chars:
        if char not in replaced_chars:
            replaced_chars[char] = True
            first_occurrence = r'(.+)'
            other_occurrences = r'\\{}'.format(len(replaced_chars))

            pattern = r'{}'.format(first_occurrence)
            output_regex = re.sub(r'{}'.format(
                char), pattern, output_regex, 1)

            pattern = r'{}'.format(other_occurrences)
            output_regex = re.sub(r'{}'.format(char), pattern, output_regex)

    return output_regex


def process_Alpha(Alpha):
    """处理公理集合 Alpha"""
    alpha = []
    for axiom in Alpha:
        processed_axiom = replace_string_with_regex(axiom)
        alpha.append(processed_axiom)
    return alpha


def check_matching(input_str, pattern_list):
    """检查输入字符串是否匹配匹配模式列表中某一个模式"""
    for pattern in pattern_list:
        if re.match(pattern, input_str):
            return True
    return False


def is_alpha_matching(input_str, pattern_list):
    """检查输入字符串是否匹配公理或定理列表中某一个模式"""
    for i, pattern in enumerate(pattern_list):
        if i == 0 and re.match(pattern, input_str):
            return "L1"
        elif i == 1 and re.match(pattern, input_str):
            return "L2"
        elif i == 2 and re.match(pattern, input_str):
            return "L3"
        elif re.match(pattern, input_str):
            return "other"
    return None


def backfill_index(label, rules, pos):
    """补充标号"""
    for i in range(pos, len(rules)):
        if '%' in rules[i]:
            rules[i] = re.sub(r'%', label, rules[i])
            break
    return rules


def add_pth(formula):
    """给公式加括号"""
    if is_need_pth(formula):
        return "(" + formula + ")"
    else:
        return formula


def insert_statement(proof_set, indexs, rules, label, expression, pos, need_backfill=False):
    """插入陈述"""
    indexs.insert(pos, len(indexs) + 1)
    if need_backfill:
        rules = backfill_index(str(len(indexs)), rules, pos)
    rules.insert(pos, label)
    proof_set.insert(pos, expression)
    return


def arrow_concat(formula1, formula2):
    """将两个公式用->连接起来"""
    return add_pth(formula1) + "->" + add_pth(formula2)


def L1(formula1, formula2):
    """L1规则"""
    return arrow_concat(formula1, arrow_concat(formula2, formula1))


def L2(formula1, formula2, formula3):
    """L2规则"""
    return arrow_concat(arrow_concat(formula1, arrow_concat(formula2, formula3)),
                        arrow_concat(arrow_concat(formula1, formula2),
                                     arrow_concat(formula1, formula3)))


def L3(formula1, formula2):
    """L3规则"""
    return arrow_concat(arrow_concat("~" + add_pth(formula1), "~" + add_pth(formula2)),
                        arrow_concat(formula2, formula1))


def inverse_permutation(perm):
    """逆置换"""
    inverse = [0] * len(perm)
    for i, p in enumerate(perm):
        inverse[p-1] = i + 1
    return inverse


def add_to_proof_set(proof_set, A, known_proof, alpha, Gamma, tuple_C, indexs, rules):
    """将符合规则的结果添加到证明集合中"""
    temp1 = arrow_concat(A, tuple_C[1])
    temp2 = is_alpha_matching(temp1, alpha)
    temp3 = is_alpha_matching(tuple_C[1], alpha)
    if temp2 or temp1 in Gamma:
        if temp1 in Gamma:
            insert_statement(proof_set, indexs, rules, "假定", temp1, 0, True)
        else:
            insert_statement(proof_set, indexs, rules, temp2, temp1, 0, True)
    elif temp3 or tuple_C[1] in Gamma:
        if tuple_C[1] in Gamma:
            insert_statement(proof_set, indexs, rules, "假定", tuple_C[1], 0)
        else:
            insert_statement(proof_set, indexs, rules, temp3, tuple_C[1], 0)
        insert_statement(proof_set, indexs, rules,
                         "L1", L1(tuple_C[1], A), 1)
        rule = "({}), ({}), MP".format(indexs[0], indexs[1])
        insert_statement(proof_set, indexs, rules, rule, temp1, 2, True)
        # proof_set.insert(0, tuple_C[1])
        # proof_set.insert(1, arrow_concat(tuple_C[1], temp1))
        # proof_set.insert(2, temp1)
    elif tuple_C[1] == A:
        temp2 = L1(A, arrow_concat(A, A))
        temp3 = L2(A, arrow_concat(A, A), A)
        _, temp4 = extract_expression_by_arrow(temp3)
        temp5 = L1(A, A)
        temp6 = arrow_concat(A, A)
        insert_statement(proof_set, indexs, rules, "L1", temp2, 0)
        insert_statement(proof_set, indexs, rules, "L2", temp3, 1)
        rule = "({}), ({}), MP".format(indexs[0], indexs[1])
        insert_statement(proof_set, indexs, rules, rule, temp4, 2)
        insert_statement(proof_set, indexs, rules, "L1", temp5, 3)
        rule = "({}), ({}), MP".format(indexs[2], indexs[3])
        insert_statement(proof_set, indexs, rules, rule, temp6, 4, True)
        # proof_set.insert(0, temp2)
        # proof_set.insert(1, temp3)
        # proof_set.insert(2, temp4)
        # proof_set.insert(3, temp5)
        # proof_set.insert(4, temp6)
    else:
        temp1, temp2 = extract_expression_by_arrow(
            known_proof[tuple_C[0]-1])
        if is_arrow_concat(known_proof[tuple_C[0]-1]) and temp2 == tuple_C[1]:
            temp_tuple2 = (tuple_C[0]-1, known_proof[tuple_C[0]-1])
            temp_tuple1 = ()
            for i, expression in enumerate(known_proof):
                if expression == temp1:
                    temp_tuple1 += (i,)
                    temp_tuple1 += (expression,)
                    break
            temp3 = L2(A, temp1, temp2)
            _, temp4 = extract_expression_by_arrow(temp3)
            _, temp5 = extract_expression_by_arrow(temp4)
            insert_statement(proof_set, indexs, rules, "L2", temp3, 0)
            rule = "(%), ({}), MP".format(indexs[0])
            insert_statement(proof_set, indexs, rules, rule, temp4, 1)
            rule = "(%), ({}), MP".format(indexs[1])
            insert_statement(proof_set, indexs, rules,
                             rule, temp5, 2, True)
            # proof_set.insert(0, temp3)
            # proof_set.insert(1, temp4)
            # proof_set.insert(2, temp5)
            # 回填要求先递归 Cj
            add_to_proof_set(proof_set, A, known_proof,
                             alpha, Gamma, temp_tuple2, indexs, rules)
            add_to_proof_set(proof_set, A, known_proof,
                             alpha, Gamma, temp_tuple1, indexs, rules)
        else:
            temp1 = arrow_concat(known_proof[tuple_C[0]-1], tuple_C[1])
            temp_tuple1 = (tuple_C[0]-1, known_proof[tuple_C[0]-1])
            temp_tuple2 = ()
            for i, expression in enumerate(known_proof):
                if expression == temp1:
                    temp_tuple2 += (i,)
                    temp_tuple2 += (expression,)
                    break
            temp2 = L2(A, known_proof[tuple_C[0]-1], tuple_C[1])
            _, temp3 = extract_expression_by_arrow(temp2)
            _, temp4 = extract_expression_by_arrow(temp3)
            insert_statement(proof_set, indexs, rules, "L2", temp2, 0)
            rule = "(%), ({}), MP".format(indexs[0])
            insert_statement(proof_set, indexs, rules, rule, temp3, 1)
            rule = "(%), ({}), MP".format(indexs[1])
            insert_statement(proof_set, indexs, rules, rule, temp4, 2, True)
            # proof_set.insert(0, temp2)
            # proof_set.insert(1, temp3)
            # proof_set.insert(2, temp4)
            # 回填要求先递归 Cj
            add_to_proof_set(proof_set, A, known_proof, alpha,
                             Gamma, temp_tuple2, indexs, rules)
            add_to_proof_set(proof_set, A, known_proof, alpha,
                             Gamma, temp_tuple1, indexs, rules)
    return


def main():
    '''主函数'''

    '''所有已知条件'''
    Gamma = input("请输入已知集合Gama，没有请回车: ").split()
    known_proof = input("请输入演绎定理证明: ").split()
    gamma = input("请输入待证公式gama: ")
    # 公理集
    Alpha = ["A->(B->A)", "(A->(B->C))->((A->B)->(A->C))",
             "(~A->~B)->(B->A)"]
    axiom_input = input("请输入要额外添加的定理，没有请回车: ").split()
    if axiom_input:
        Alpha.extend(axiom_input)
    # 公理集的模式匹配集合
    alpha = process_Alpha(Alpha)

    '''解析待证公式'''
    if is_arrow_concat(gamma):
        A, B = extract_expression_by_arrow(gamma)
    else:
        print("输入的待证公式不符合规范")
        return
    proof_set = []
    indexs = []
    rules = []
    add_to_proof_set(proof_set, A, known_proof, alpha, Gamma,
                     (len(known_proof)-1, B), indexs, rules)

    indexs = inverse_permutation(indexs)

    for i in range(len(rules)):
        if 'MP' in rules[i]:
            nums = re.findall(r'\d+', rules[i])
            for num in nums:
                rules[i] = re.sub(r'{}'.format(num), r'{}'.format(
                    indexs[int(num)-1]), rules[i])

    # 将公式中的 -> 替换为 &rightarrow;，优化 markdown 表格
    # for i in range(len(proof_set)):
    #     proof_set[i] = re.sub(r'->', r' &rightarrow; ', proof_set[i])

    # 生成 markdown 表格
    with open("proof_set.txt", "w", encoding='utf-8') as file:
        file.write("|序号|公式|规则|\n")
        file.write("|---|---|---|\n")
        for i in range(len(proof_set)):
            file.write('|' + '({})'.format(i+1) + '|' +
                       proof_set[i] + '|' + rules[i] + '|' + "\n")


if __name__ == "__main__":
    main()
