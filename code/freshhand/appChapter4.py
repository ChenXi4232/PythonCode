list1 = ['first', 'second', 'third']
list2 = list1
num1 = 1
num2 = num1
list2.append('fourth')
num2 = 1 + 1
print(list1)
print(num1)
# python列表变量赋值变量并非新开辟一个存储空间，而是将两个变量名关联，修改其中一个变量指向内容，另一个变量值同步变化

list3 = (1, 2)
print(list3[0])
# 元组只能为数字

list5 = ['first', '2', 3, 'fourth']
for element in list5[:]:
    print(element)
print(list[1])  # ?????
print(list5[2])
