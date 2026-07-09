import random
import matplotlib.pyplot as plt

# ---------- 编码解码函数 ----------
def encode(x):
    if not (1 <= x < 4):
        raise ValueError("x must be in [1, 4)")
    int_part = int(x)
    dec_part = round(x - int_part, 2)
    tenth = int(dec_part * 10)
    hundredth = int(round(dec_part * 100 - tenth * 10))
    int_bin = format(int_part, '02b')
    tenth_bin = format(tenth, '04b')
    hundredth_bin = format(hundredth, '04b')
    return int_bin + tenth_bin + hundredth_bin

def decode(binary_str):
    int_part = int(binary_str[0:2], 2)
    tenth = int(binary_str[2:6], 2)
    hundredth = int(binary_str[6:9], 2)
    if tenth > 9:
        tenth = 9
    if hundredth > 9:
        hundredth = 9
    x = int_part + tenth / 10.0 + hundredth / 100.0
    return max(1.0, min(3.99, x))

# ---------- 交叉和变异 ----------
def crossover(parent1_bin, parent2_bin):
    point = random.randint(1, len(parent1_bin)-1)
    child1 = parent1_bin[:point] + parent2_bin[point:]
    child2 = parent2_bin[:point] + parent1_bin[point:]
    return child1, child2

def mutate(binary_str, pm):
    mutant = ""
    for bit in binary_str:
        if random.random() < pm:
            mutant += '1' if bit == '0' else '0'
        else:
            mutant += bit
    return mutant

# ---------- 适应度函数 ----------
def fitness(x):
    return 1.0 / ((x - 3)**2 + 1e-6)

# ---------- 选择（锦标赛选择） ----------
def tournament_selection(population, fitnesses, k=2):
    selected = []
    for _ in range(len(population)):
        indices = random.sample(range(len(population)), k)
        best_idx = max(indices, key=lambda i: fitnesses[i])
        selected.append(population[best_idx])
    return selected

# ---------- 主算法 ----------
def evolutionary_algorithm(Gen, pc, pm, P0, P, target_func):
    population = []
    for _ in range(P0):
        x = random.uniform(1.0, 3.99)
        population.append(encode(x))

    best_x_history = []

    for gen in range(Gen):
        decoded_pop = [decode(ind) for ind in population]
        fitnesses = [fitness(x) for x in decoded_pop]

        best_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
        best_x = decoded_pop[best_idx]
        best_x_history.append(best_x)

        new_population = population.copy()
        while len(new_population) < P:
            p1 = random.choice(population)
            p2 = random.choice(population)
            if random.random() < pc:
                c1, c2 = crossover(p1, p2)
                new_population.append(c1)
                new_population.append(c2)
            else:
                new_population.append(p1)
                new_population.append(p2)

        for i in range(len(new_population)):
            if random.random() < pm:
                new_population[i] = mutate(new_population[i], pm)

        decoded_new = [decode(ind) for ind in new_population]
        fitnesses_new = [fitness(x) for x in decoded_new]
        population = tournament_selection(new_population, fitnesses_new, k=2)

    final_decoded = [decode(ind) for ind in population]
    final_fitnesses = [fitness(x) for x in final_decoded]
    best_idx_final = max(range(len(final_fitnesses)), key=lambda i: final_fitnesses[i])
    best_x_final = final_decoded[best_idx_final]
    best_y_final = target_func(best_x_final)

    return best_x_final, best_y_final, best_x_history

# 目标函数
def target_func(x):
    return (x - 3)**2

# ---------- 运行不同参数设置 ----------
params_sets = [
    {"Gen": 10, "pc": 0.6, "pm": 0.2, "P0": 10, "P": 15},
    {"Gen": 20, "pc": 0.6, "pm": 0.2, "P0": 10, "P": 15},
    {"Gen": 20, "pc": 0.6, "pm": 0.2, "P0": 20, "P": 30},
    {"Gen": 20, "pc": 0.6, "pm": 0.2, "P0": 5, "P": 7},
    {"Gen": 20, "pc": 0.2, "pm": 0.2, "P0": 10, "P": 15},
]

results = []
for i, params in enumerate(params_sets):
    best_x, best_y, hist_x = evolutionary_algorithm(
        params["Gen"], params["pc"], params["pm"], params["P0"], params["P"], target_func
    )
    results.append({
        "params": params,
        "best_x": best_x,
        "best_y": best_y,
        "history": hist_x
    })
    print(f"参数 {i+1}: Gen={params['Gen']}, pc={params['pc']}, pm={params['pm']}, P0={params['P0']}, P={params['P']}")
    print(f"  最优解: x = {best_x:.4f}, y = {best_y:.6f}\n")

# ---------- 绘制收敛曲线（保存 + 弹窗） ----------
plt.figure(figsize=(12, 8))
for i, res in enumerate(results):
    plt.plot(res["history"], label=f"Param {i+1}", marker='o', markersize=3)
plt.xlabel("Generation")
plt.ylabel("Best x value")
plt.title("Convergence of Best Solution under Different Parameter Settings")
plt.legend()
plt.grid(True)

# 保存图片到当前文件夹
plt.savefig('convergence_curve.png', dpi=300, bbox_inches='tight')
print("✅ 收敛曲线图已保存为: convergence_curve.png")

# 显示图片弹窗
plt.show()