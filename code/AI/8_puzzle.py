import random
import heapq

# 定义目标状态
goal_state = [1, 2, 3, 4, 5, 6, 7, 8, 0]


# 定义启发式函数（曼哈顿距离）
def std_heuristic(state):
    distance = 0
    for i in range(9):
        if state[i] == 0:
            continue
        goal_row = (state[i] - 1) // 3
        goal_col = (state[i] - 1) % 3
        current_row = i // 3
        current_col = i % 3
        distance += abs(goal_row - current_row) + abs(goal_col - current_col)
    return distance


# 定义启发式函数（曼哈顿距离+错误放置的方块数）
def ovestm_heuristic(state):
    manhattan_distance = 0
    misplaced_tiles = 0
    for i in range(9):
        if state[i] == 0:
            continue
        goal_row = (state[i] - 1) // 3
        goal_col = (state[i] - 1) % 3
        current_row = i // 3
        current_col = i % 3
        manhattan_distance += abs(goal_row - current_row) + \
            abs(goal_col - current_col)
        if state[i] != goal_state[i]:
            misplaced_tiles += 1
    return manhattan_distance + misplaced_tiles


# 定义移动操作
def move(state, direction):
    new_state = state.copy()
    zero_index = new_state.index(0)
    if direction == 'up' and zero_index not in [0, 1, 2]:
        new_state[zero_index], new_state[zero_index -
                                         3] = new_state[zero_index - 3], new_state[zero_index]
    elif direction == 'down' and zero_index not in [6, 7, 8]:
        new_state[zero_index], new_state[zero_index +
                                         3] = new_state[zero_index + 3], new_state[zero_index]
    elif direction == 'left' and zero_index not in [0, 3, 6]:
        new_state[zero_index], new_state[zero_index -
                                         1] = new_state[zero_index - 1], new_state[zero_index]
    elif direction == 'right' and zero_index not in [2, 5, 8]:
        new_state[zero_index], new_state[zero_index +
                                         1] = new_state[zero_index + 1], new_state[zero_index]
    return new_state


# A*搜索算法
def astar(start_state, heuristic=std_heuristic):
    open_list = []
    closed_set = set()
    heapq.heappush(open_list, (heuristic(start_state), start_state, []))

    while open_list:
        _, current_state, path = heapq.heappop(open_list)

        if current_state == goal_state:
            return path

        if tuple(current_state) in closed_set:
            continue

        closed_set.add(tuple(current_state))

        zero_index = current_state.index(0)
        row, col = zero_index // 3, zero_index % 3

        for direction in ['up', 'down', 'left', 'right']:
            if (direction == 'up' and row == 0) or (direction == 'down' and row == 2) or \
               (direction == 'left' and col == 0) or (direction == 'right' and col == 2):
                continue

            new_state = move(current_state, direction)
            new_path = path + [direction]
            heapq.heappush(open_list, (len(new_path) +
                           heuristic(new_state), new_state, new_path))

    return None


# 打印九宫格
def print_grid(grid):
    for i in range(3):
        row = grid[i*3:i*3+3]
        print(" ".join(map(str, row)))


# 测试
while True:
    # 打印目标状态
    print('Goal State:')
    print('------------------')
    print_grid(goal_state)
    print('------------------')
    # 生成长度为九的随机列表
    start_state = random.sample(range(9), 9)
    print('Start State:')
    print('------------------')
    print_grid(start_state)
    print('------------------')

    std_result = astar(start_state)
    ovestm_result = astar(start_state, ovestm_heuristic)

    if std_result is None or ovestm_result is None:
        continue
    elif len(std_result) != len(ovestm_result):
        print('best result = {}, path:{}'.format(len(std_result), std_result))
        print('overestimated result = {}, overestimated path:{}'.format(
            len(ovestm_result), ovestm_result))
        break
