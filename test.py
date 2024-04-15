from pulp import LpProblem, LpMaximize, LpVariable, lpSum, value

num_products = 3
num_markets = 3

prices = [[10, 9, 8], [9, 8, 7], [8, 7, 6]]

products = [chr(65+i) for i in range(num_products)]
markets = range(1, num_markets+1)
shifts = range(1, 5)

# 产品数量变量
production_vars = LpVariable.dicts(
    "Production", (products, shifts), lowBound=0, cat='Continuous')

# 目标函数
avg_prices = []
for p in products:
    avg_price = 0
    for m in markets:
        avg_price += prices[m-1][products.index(p)]
    avg_price /= num_markets
    avg_prices.append(avg_price)

expr = lpSum([avg_prices[i] * production_vars[products[i]][s]
             for i in range(num_products) for s in shifts])

print(expr)

wages = [10, 9, 8, 7]

shifts_times = [8, 4, 8, 4]
shifts_wages = [wages[i] * shifts_times[i] for i in range(4)]
shifts_wages = [shifts_wages[0]] + shifts_wages

total_employees = LpVariable.dicts(
    "E", range(0, 5), lowBound=0, cat='Continuous')
total_wages = [lpSum([total_employees[s] * shifts_wages[s] for s in range(5)])]
print(total_wages)
