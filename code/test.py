def count_nums(h, w):
    if h > w:
        return count_nums(h-w, w) + 1
    elif h < w:
        return count_nums(h, w-h) + 1
    else:
        return 1


print(count_nums(2019, 324))
