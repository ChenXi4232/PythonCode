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


def add_pth(formula):
    """给公式加括号"""
    if is_need_pth(formula):
        return "(" + formula + ")"
    else:
        return formula


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


def add_to_proof_set(proof_set, A, known_proof, alpha, Gamma, tuple_C):
    """将符合规则的结果添加到证明集合中"""
    temp1 = arrow_concat(A, tuple_C[1])
    if check_matching(temp1, alpha) or temp1 in Gamma:
        proof_set.insert(0, emp1)
        print(proof_set[-1])
    elif check_matching(tuple_C[1], alpha) or tuple_C[1] in Gamma:
        proof_set.insert(0, tuple_C[1])
        print(proof_set[-1])
        proof_set.insert(1, arrow_concat(tuple_C[1], temp1))
        print(proof_set[-1])
        proof_set.insert(2, temp1)
        print(proof_set[-1])
    elif tuple_C[1] == A:
        temp2 = L1(A, arrow_concat(known_proof[-1], A))
        temp3 = L2(A, arrow_concat(known_proof[-1], A), A)
        _, temp4 = extract_expression_by_arrow(temp3)
        temp5 = L1(A, known_proof[-1])
        temp6 = arrow_concat(A, A)
        proof_set.insert(0, temp2)
        print(proof_set[-1])
        proof_set.insert(1, temp3)
        print(proof_set[-1])
        proof_set.insert(2, temp4)
        print(proof_set[-1])
        proof_set.insert(3, temp5)
        print(proof_set[-1])
        proof_set.insert(4, temp6)
        print(proof_set[-1])
    else:
        if is_arrow_concat(known_proof[tuple_C[0]-1]):
            temp1, temp2 = extract_expression_by_arrow(
                known_proof[tuple_C[0]-1])
            if temp2 == tuple_C[1]:
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
                proof_set.insert(0, temp3)
                print(proof_set[-1])
                proof_set.insert(1, temp4)
                print(proof_set[-1])
                proof_set.insert(2, temp5)
                print(proof_set[-1])
                add_to_proof_set(proof_set, A, known_proof,
                                 alpha, Gamma, temp_tuple1)
                add_to_proof_set(proof_set, A, known_proof,
                                 alpha, Gamma, temp_tuple2)
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
            proof_set.insert(0, temp2)
            print(proof_set[-1])
            proof_set.insert(1, temp3)
            print(proof_set[-1])
            proof_set.insert(2, temp4)
            print(proof_set[-1])
            add_to_proof_set(proof_set, A, known_proof,
                             alpha, Gamma, temp_tuple1)
            add_to_proof_set(proof_set, A, known_proof,
                             alpha, Gamma, temp_tuple2)
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
    add_to_proof_set(
        proof_set, A, known_proof, alpha, Gamma, (len(known_proof)-1, B))
    with open("proof_set.txt", "w") as file:
        for expression in proof_set:
            file.write(expression + "\n")


if __name__ == "__main__":
    main()
