import numpy as np
import matplotlib.pyplot as plt


class PSO(object):
    def __init__(self, population_size, max_steps):

        self.w = 0.6  # 惯性权重
        self.c1 = self.c2 = 1  # 加速系数
        self.population_size = population_size  # 粒子群数量
        self.dim = 3  # 搜索空间的维度 3表示三维空间
        self.max_steps = max_steps  # 迭代次数
        self.x_bound = [-20, 20]  # 解空间范围

        # 初始化粒子群位置
        self.x = np.random.uniform(self.x_bound[0], self.x_bound[1], (self.population_size, self.dim))

        # 初始化粒子群速度
        self.v = np.random.rand(self.population_size, self.dim)

        # self.x 保存了粒子群的位置，fitness 保存 粒子群位置每一行的平方和
        fitness = self.calculate_fitness(self.x)
        self.p = self.x  # 个体的最佳位置

        # np.argmin(fitness) 计算出fitness中的最小值下标
        self.pg = self.x[np.argmin(fitness)]  # 全局最佳位置

        self.individual_best_fitness = fitness  # 个体的最优适应度

        self.global_best_fitness = np.min(fitness)  # 全局最佳适应度

    def calculate_fitness(self, x):
        # square 计算出矩阵中每个数的平方，sum(矩阵, axis=1), 二维数组的话，把每一行加起来，变成一维数组
        return np.sum(np.square(x), axis=1)

    def evolve(self):

        # max_steps 迭代次数
        for step in range(self.max_steps):
            # 生成行为population_size，列为dim的二维数组
            r1 = np.random.rand(self.population_size, self.dim)
            r2 = np.random.rand(self.population_size, self.dim)
            # 更新速度和权重
            self.v = self.w * self.v + self.c1 * r1 * (self.p - self.x) + self.c2 * r2 * (self.pg - self.x)
            self.x = self.v + self.x
            plt.clf()
            plt.scatter(self.x[:, 0], self.x[:, 1], s=30, color='k')
            plt.xlim(self.x_bound[0], self.x_bound[1])
            plt.ylim(self.x_bound[0], self.x_bound[1])
            plt.pause(0.01)
            fitness = self.calculate_fitness(self.x)
            # 需要更新的个体
            update_id = np.greater(self.individual_best_fitness, fitness)
            self.p[update_id] = self.x[update_id]
            self.individual_best_fitness[update_id] = fitness[update_id]
            # 新一代出现了更小的fitness，所以更新全局最优fitness和位置
            if np.min(fitness) < self.global_best_fitness:
                self.pg = self.x[np.argmin(fitness)]
                self.global_best_fitness = np.min(fitness)
            print('best fitness: %.5f, mean fitness: %.5f' % (self.global_best_fitness, np.mean(fitness)))


pso = PSO(200, 300)
pso.evolve()
plt.show()
