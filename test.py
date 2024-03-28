def KMP(S, T):
    nextval_list = get_nextval(T)
    match_positions = []
    i, j = 0, 0
    S_length, T_length = len(S), len(T)
    while i < S_length:
        if S[i] == T[j]:
            i += 1
            j += 1
        if j == T_length:
            match_positions.append(i - len(T))
            j = nextval_list[j-1]
        elif i < S_length and S[i] != T[j]:
            if j != 0:
                j = nextval_list[j-1]
            else:
                i += 1
    return match_positions


def get_nextval(T):
    T_length = len(T)
    nextval_list = [0] * T_length
    nextval_list[0] = 0
    i, j = 1, 0
    while i < T_length:
        if T[i] == T[j]:
            j += 1
            nextval_list[i] = j
            i += 1
        else:
            if j != 0:
                j = nextval_list[j-1]
            else:
                nextval_list[i] = 0
                i += 1
    return nextval_list


'''def get_nextval(T):
    T_length = len(T)
    nextval_list = [-1] * T_length
    nextval_list[0] = -1
    i, j = 0, -1
    while i < T_length - 1:
        if j == -1 or T[i] == T[j]:
            i += 1
            j += 1
            if T[i] != T[j]:
                nextval_list[i] = j
            else:
                nextval_list[i] = nextval_list[j]
        else:
            i = nextval_list[i]
    return nextval_list'''


S_length, T_length = map(int, input().split())
S = input()
T = input()

match_positions = KMP(S, T)
num_matches = len(match_positions)
print(num_matches)
print(*match_positions)
