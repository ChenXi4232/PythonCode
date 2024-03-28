import random
import time
import copy
import sys
import numpy as np

num = 10000000
coll = [[0] * 2] * (num * 2)
coll = np.array(coll)


def count_conflicts(q, n, i, j, c):
    conflict = c
    #删除原有冲突
    coll[q[i] + i][0] -= 1
    if (coll[q[i] + i][0] > 0):
        conflict -= 1
    coll[q[i] - i + n - 1][1] -= 1
    if (coll[q[i] - i + n - 1][1] > 0):
        conflict -= 1
    coll[q[j] + j][0] -= 1
    if (coll[q[j] + j][0] > 0):
        conflict -= 1
    coll[q[j] - j + n - 1][1] -= 1
    if (coll[q[j] - j + n - 1][1] > 0):
        conflict -= 1
    #现有冲突
    coll[q[i] + j][0] += 1
    if (coll[q[i] + j][0] > 1):
        conflict += 1
    coll[q[i] - j + n - 1][1] += 1
    if (coll[q[i] - j + n - 1][1] > 1):
        conflict += 1
    coll[q[j] + i][0] += 1
    if (coll[q[j] + i][0] > 1):
        conflict += 1
    coll[q[j] - i + n - 1][1] += 1
    if (coll[q[i] - j + n - 1][1] > 1):
        conflict += 1
    return conflict


def recover_conflicts(q, n, i, j):
    coll[q[i] + i][0] += 1
    coll[q[i] - i + n - 1][1] += 1
    coll[q[j] + j][0] += 1
    coll[q[j] - j + n - 1][1] += 1
    coll[q[i] + j][0] -= 1
    coll[q[i] - j + n - 1][1] -= 1
    coll[q[j] + i][0] -= 1
    coll[q[j] - i + n - 1][1] -= 1


def count_all_conflicts(q, n):
    conflict = 0
    for i in range(n):
        coll[q[i] + i][0] += 1
        coll[q[i] - i + n - 1][1] += 1
    for i in range(n * 2):
        for j in range(0, 2):
            if coll[i][j] > 1:
                # print(i,j)
                conflict += coll[i][j] - 1
    return conflict


if __name__ == "__main__":
    q = random.sample(range(0, num), num)
    min_conflicts = count_all_conflicts(q, num)
    start = time.time()
    # print(coll)
    while (min_conflicts > 0):
        # print(q)
        print(min_conflicts)
        if (min_conflicts == 0):
            break
        flag = 0
        for i in range(num):
            for j in range(i + 1, num):
                # conflicts_before = count_conflicts(q,num,i,j)
                # q[i],q[j]=q[j],q[i]
                conflicts = count_conflicts(q, num, i, j, min_conflicts)
                # print(min_conflicts,q)
                if (conflicts < min_conflicts):
                    q[i], q[j] = q[j], q[i]
                    min_conflicts = conflicts
                    flag = 1
                    break
                else:
                    recover_conflicts(q, num, i, j)
            if (flag == 1):
                break
        if (flag == 0):
            q = random.sample(range(0, num), num)
            min_conflicts = count_all_conflicts(q, num)
    print('queen:', q)
    end = time.time()
    print('time', end - start)
