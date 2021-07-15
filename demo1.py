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
        print(self.individual_best_fitness)
        self.global_best_fitness = np.min(fitness)  # 全局最佳适应度

    # 重点是这个函数的功能，还没理解
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
            plt.close()

            # plt.scatter
            # 画散点图的方法，self.x是一个二维数组，数组的每个位置都是一个含有三个数(x, y, z)的数组，也就是每个粒子当前在空间中的位置
            # self.x[:, 0] 的含义就是取到所有粒子的横坐标
            # self.x[:, 1] 的含义就是取到所有粒子的纵坐标
            # 参数 s 的目前我理解是控制了粒子在图中的大小
            # c 即 color 表示粒子的颜色
            plt.scatter(self.x[:, 0], self.x[:, 1], s=30, c='r')

            # plt.xlim(xmin, xmax)
            # 设置 x 轴的显示范围
            # xmin 最小值即左边的数
            # xmax 最大值即右边的数
            plt.xlim(self.x_bound[0], self.x_bound[1])

            # plt.ylim(ymin, ymax)
            # 设置 y 轴的显示范围
            # ymin 最小值即下边的数
            # ymax 最大值即上边的数
            plt.ylim(self.x_bound[0], self.x_bound[1])

            # plt.pause(time)
            # 当前理解：图像窗口停留的时间，传入 1 就是停留 1 秒
            plt.pause(1)

            #
            fitness = self.calculate_fitness(self.x)

            #
            # 需要更新的个体
            # update_id 保存了一列包含 True, False 的一维数组
            update_id = np.greater(self.individual_best_fitness, fitness)
            # print("self.x = ", self.x)
            # print("update_id = ", update_id)
            # print("self.x[update_id] = ", self.x[update_id])
            # self.x[update_id]中 在 update_id 中对应位置为 True 的数会被保存下来

            self.p[update_id] = self.x[update_id]

            # print("===========")
            # print("fitness = ", fitness)
            # print("fitness[update_id] = ", fitness[update_id])

            # fitness[update_id]中 在 update_id 中对应位置为 True 的数会被保存下来
            self.individual_best_fitness[update_id] = fitness[update_id]

            # 新一代出现了更小的fitness，所以更新全局最优fitness和位置
            print("np.min(fitness) = ", np.min(fitness))
            print("self.global_best_fitness", self.global_best_fitness)
            if np.min(fitness) < self.global_best_fitness:
                self.pg = self.x[np.argmin(fitness)]
                self.global_best_fitness = np.min(fitness)
            # best fitness 最佳适应度，mean fitness 平均适应度
            print('best fitness: %.5f, mean fitness: %.5f' % (self.global_best_fitness, np.mean(fitness)))


pso = PSO(5, 20)
pso.evolve()
plt.show()
