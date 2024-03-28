bicycles = ["trek", "cannondale", "redline", "specialized"]
print(bicycles)

for bicycle in bicycles:
    print(bicycle)

for num in range(0, 4, 1):
    print(bicycles[num].title())

bicycles.append('ducati')
print(bicycles)

word = bicycles.remove('ducati')
print(bicycles)

bicycles.insert(1, 'ducati')
print(bicycles)

Bicycles = bicycles
print(Bicycles)

names = ['wang chen', 'zhang junpei', 'du daiyang']
Bicycles = names
for name in Bicycles:
    print(name.title())

names = ['wang chen', 'zhang junpei', 'du daiyang', 'wang chen']
print(names)
name = names.remove('wang chen')
print(names)
print(name)

names = ['wang chen', 'zhang junpei', 'du daiyang', 'wang chen']
name = names.pop(2)
print(name)
print(names)

names = ['wang chen', 'zhang junpei', 'du daiyang', 'wang chen']
del names[2]
for name in names:
    print(name)
# pop可以赋值 pop和del必须使用下标索引 remove无法赋值且不报错

names = bicycles.sort()
print(names)
names = bicycles.sort(reverse=True)
print(names)

bicycles = ["trek", "cannondale", "redline", "specialized"]
names = sorted(bicycles)
print(names)
print(bicycles)
names.reverse()
print(names.reverse())
print(names)
# 方法sort无法赋值，但函数sorted可以 方法reverse不能在print里使用

print(len(names))
