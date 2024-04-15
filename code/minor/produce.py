from pulp import LpProblem, LpMaximize, LpVariable, lpSum, value


def input_parameters_from_terminal():
    # 输入产品数
    num_products = int(input("请输入产品数："))
    # 输入市场数
    num_markets = int(input("请输入市场数："))

    # 输入每个市场上每个产品的价格
    prices = []
    for i in range(num_markets):
        market_prices = []
        print(f"请输入市场{i+1}上每个产品的价格：")
        for j in range(num_products):
            price = float(input(f"产品{j+1}的价格："))
            market_prices.append(price)
        prices.append(market_prices)

    # 输入员工总数
    total_employees = int(input("请输入员工总数："))
    # 输入机器总数
    total_machines = int(input("请输入机器总数："))

    # 输入每个产品最低预期生产数量
    min_production_quantities = []
    for i in range(num_products):
        quantity = int(input(f"请输入产品{i+1}的最低预期生产数量："))
        min_production_quantities.append(quantity)

    # 输入每个产品生产需要的机器工时
    machine_hours = []
    for i in range(num_products):
        hours = float(input(f"请输入产品{i+1}的生产需要的机器工时："))
        machine_hours.append(hours)

    # 输入每个产品生产需要的人力工时
    labor_hours = []
    for i in range(num_products):
        hours = float(input(f"请输入产品{i+1}的生产需要的人力工时："))
        labor_hours.append(hours)

    # 输入每个产品的原材料费
    material_costs = []
    for i in range(num_products):
        cost = float(input(f"请输入产品{i+1}的原材料费："))
        material_costs.append(cost)

    # 输入四个班次工人的工资
    wages = []
    for i in range(4):
        wage = float(input(f"请输入第{i+1}个班次工人的工资："))
        wages.append(wage)

    # 将参数写入到文件
    with open('parameters.txt', 'w') as file:
        file.write(f"产品数：{num_products}\n")
        file.write(f"市场数：{num_markets}\n")
        file.write(f"机器工时：{machine_hours}\n")
        file.write(f"人力工时：{labor_hours}\n")
        file.write(f"原材料费：{material_costs}\n")
        file.write(f"工资：{wages}\n")

    return num_products, num_markets, prices, total_employees, total_machines, min_production_quantities, machine_hours, labor_hours, material_costs, wages


def input_parameters_from_file():
    with open('parameters.txt', 'r') as file:
        lines = file.readlines()
        num_products = int(lines[0].split('：')[1])
        num_markets = int(lines[1].split('：')[1])
        machine_hours = list(
            map(float, lines[2].split('：')[1][1:-2].split(',')))
        labor_hours = list(map(float, lines[3].split('：')[1][1:-2].split(',')))
        material_costs = list(
            map(float, lines[4].split('：')[1][1:-2].split(',')))
        wages = list(
            map(float, lines[4].split('：')[1][1:-2].split(',')))

    # 输入员工总数
    total_employees = int(input("请输入员工总数："))
    # 输入机器总数
    total_machines = int(input("请输入机器总数："))

    # 输入每个市场上每个产品的价格
    prices = []
    for i in range(num_markets):
        market_prices = []
        print(f"请输入市场{i+1}上每个产品的价格：")
        for j in range(num_products):
            price = float(input(f"产品{j+1}的价格："))
            market_prices.append(price)
        prices.append(market_prices)

    # 输入每个产品最低预期生产数量
    min_production_quantities = []
    for i in range(num_products):
        quantity = int(input(f"请输入产品{i+1}的最低预期生产数量："))
        min_production_quantities.append(quantity)

    return num_products, num_markets, prices, total_employees, total_machines, min_production_quantities, machine_hours, labor_hours, material_costs, wages


def solve_production_problem(num_products, num_markets, machine_hours, labor_hours, material_costs, wages, min_production_quantities, total_machines, prices):
    # 创建问题
    prob = LpProblem("Production Optimization", LpMaximize)

    # 创建变量
    products = [chr(65+i) for i in range(num_products)]
    markets = range(1, num_markets+1)
    shifts = range(1, 5)
    shifts_times = [8, 4, 8, 4]
    shifts_wages = [wages[i] * shifts_times[i] for i in range(4)]
    shifts_wages = [shifts_wages[0]] + shifts_wages

    # 产品数量变量
    production_vars = LpVariable.dicts(
        "Production", (products, shifts), lowBound=0, cat='Continuous')

    # 人工数量变量
    total_employees = LpVariable.dicts(
        "E", range(5), lowBound=0, cat='Continuous')

    # 机器数量变量
    total_machines = LpVariable.dicts(
        "M", shifts, lowBound=0, cat='Continuous')

    total_wages = [lpSum([total_employees[s] * shifts_wages[s]
                         for s in range(5)])]

    # 目标函数
    avg_prices = []
    for p in products:
        avg_price = 0
        for m in markets:
            avg_price += prices[m-1][products.index(p)]
        avg_price /= num_markets
        avg_prices.append(avg_price)
    prob += lpSum([avg_prices[i] * production_vars[products[i]][s] for i in range(num_products) for s in shifts]) \
        - 65 * lpSum(total_wages) - lpSum(material_costs)

    # 添加约束
    for p in products:
        for m in markets:
            prob += production_vars[p][m][1] + \
                production_vars[p][m][2] == min_production_quantities[products.index(
                    p)], f"MinProduction_{p}{m}"
            prob += lpSum([machine_hours[products.index(p)] * production_vars[p][m][s]
                          for s in shifts]) == total_machines, f"MachineHours_{p}{m}"
            prob += lpSum([labor_hours[products.index(p)] * production_vars[p][m][s]
                          for s in shifts]) == total_employees, f"LaborHours_{p}{m}"

    # 求解问题
    prob.solve()

    # 输出结果
    for v in prob.variables():
        print(v.name, "=", v.varValue)

    print("Total Profit =", value(prob.objective))


def main():
    input_option = input("请选择输入方式（终端输入-1，文件输入-2）：")
    if input_option == '1':
        num_products, num_markets, prices, total_employees, total_machines, min_production_quantities, machine_hours, labor_hours, material_costs, wages = input_parameters_from_terminal()
    elif input_option == '2':
        num_products, num_markets, prices, total_employees, total_machines, min_production_quantities, machine_hours, labor_hours, material_costs, wages = input_parameters_from_file()
        print(f"产品数：{num_products}")
        print(f"市场数：{num_markets}")
        print(f"机器工时：{machine_hours}")
        print(f"人力工时：{labor_hours}")
        print(f"原材料费：{material_costs}")
        print(f"工资：{wages}")
        print(f"员工总数：{total_employees}")
        print(f"机器总数：{total_machines}")
        print(f"每个市场上每个产品的价格：{prices}")
        print(f"每个产品最低预期生产数量：{min_production_quantities}")
    else:
        print("请选择正确的输入方式！")
        return

    # 使用之前的输入参数
    num_products, num_markets, prices, total_employees, total_machines, min_production_quantities, machine_hours, labor_hours, material_costs, wages = input_parameters_from_terminal()

    # 调用函数解决问题
    solve_production_problem(num_products, num_markets, machine_hours, labor_hours,
                             material_costs, wages, min_production_quantities, total_employees, total_machines)

    # 这里可以继续编写你的程序，使用这些参数进行计算或者其他操作。


if __name__ == "__main__":
    main()
