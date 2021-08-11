
# 详情参考 https://zhuanlan.zhihu.com/p/148017730

"""
代码思路：
    1.  首先初始化了粒子群自身相关的一些属性(变量)，例如惯性权重，加速系数，粒子群数量等，
        和一些客观属性(变量)，如空间维度，迭代次数，解空间范围等
    2.

"""

# numpy 一个对数组(列表)进行计算的库
# 封装了很多对数组进行运算的方法
# 比python自带的更高效
import numpy as np
# matplotlib.pyplot 画图库
# 通过调用其中的画图函数，进行绘图
import matplotlib.pyplot as plt


class PSO:
    """
        类的初始化方法 self 是一个指向类自身的关键字，来调用类自身的属性值或自身的方法
        后面的都是在类外部调用类时候传进来的参数
        population_size 为粒子群数量
        max_steps 为迭代次数
        在最后调用了此类，传入了两个参数，第一个是粒子群数量，第二个是迭代次数
    """
    def __init__(self, population_size, max_steps):

        self.w = 0.6  # 惯性权重
        self.c1 = self.c2 = 1  # 加速系数
        self.population_size = population_size  # 粒子群数量
        self.dim = 3  # 搜索空间的维度 3表示三维空间
        self.max_steps = max_steps  # 迭代次数
        self.x_bound = [-20, 20]  # 解空间范围

        """
            初始化粒子群位置
            np.random.uniform(low, high, size) 从一个均匀分布的[low, high)中取随机数，左闭右开
            self.x_bound[0] 为左边界，self.x_bound[1] 为右边界
            (self.population_size, self.dim) 第三个参数是一个元组(元组是不可改变的数组(列表))
            最终生成的随机数是一个二维数组的格式，二维数组的长度为 self.population_size，即粒子群个数
            二维数组中每个位置的值是一个含有 self.dim 个数的数组，上面设置的 self.dim 为3，三个数即为每个粒子的 x, y, z 空间坐标
        """
        self.x = np.random.uniform(self.x_bound[0], self.x_bound[1], (self.population_size, self.dim))

        """
            初始化粒子群速度
            np.random.rand(d0, d1, d2....dn) 生成 [0, 1) 的随机数
            不传参数，生成一个随机数
            传一个参数 d0，生成一个长度为 d0 的随机数数组
            传两个参数 d0, d1，生成一个长度为 (d0 * d1) 的二维随机数数组
            ... 传 N 个参数就生成一个 N 维随机数数组
            下面是生成了一个长度为 (sel.population_size * self.dim) 的二维随机数数组
            self.dim 确定是3，二维数组每个位置都含有 3 个数，也就是每个粒子生成了 3 个不同的速度
            我暂时理解为生成的 3 个速度即为粒子在 x, y, z 轴上的 3 个不同速度
        """
        self.v = np.random.rand(self.population_size, self.dim)

        """
            self.x 保存了粒子群的位置，fitness 保存粒子群中每个粒子的 x, y, z 坐标的平方之和(x^2 + y^2 + z^2)
            生成一个长度为 self.population_size 的一维数组，每个位置保存的就是该粒子的适应度
            --- 暂时不理解这样做平方和来当作适应度 ---
        """
        fitness = self.calculate_fitness(self.x)

        self.p = self.x  # 个体的最佳位置

        """
            np.argmin(fitness) 计算出 fitness 中的最小值下标
            也就是通过计算出上面适应度 fitness 一维数组中最小值的下标
            再通过这个下标值到 self.x 粒子群位置中找到这个适应度值最小的粒子，当作全局最佳位置
        """
        self.pg = self.x[np.argmin(fitness)]  # 全局最佳位置

        self.individual_best_fitness = fitness  # 个体的最优适应度
        # print(self.individual_best_fitness)
        self.global_best_fitness = np.min(fitness)  # 全局最佳适应度

    # 计算适应度，返回一个一维数组
    def calculate_fitness(self, x):
        """
            square 计算出矩阵中每个数的平方，sum(矩阵, axis=1)
            对于参数 axis, 且假设矩阵是 [[1, 2], [3, 4], [5, 6]]
            当 axis = 0, 返回结果 [9, 12]
            当 axis = 1, 返回结果 [3, 7, 11]
            :param x: 当前粒子群位置
            :return: 粒子群适应度
        """
        return np.sum(np.square(x), axis=1)

    def evolve(self):
        # max_steps 迭代次数
        for step in range(self.max_steps):

            # 生成 population_size * dim 的二维数组
            r1 = np.random.rand(self.population_size, self.dim)
            r2 = np.random.rand(self.population_size, self.dim)

            # 更新速度和粒子群的位置
            self.v = self.w * self.v + self.c1 * r1 * (self.p - self.x) + self.c2 * r2 * (self.pg - self.x)
            self.x = self.v + self.x

            # 关闭图形窗口
            plt.close()

            """
                plt.scatter
                画散点图的方法，self.x是一个二维数组，数组的每个位置都是一个含有三个数(x, y, z)的数组，也就是每个粒子当前在空间中的坐标
                self.x[:, 0] 的含义就是取到所有粒子的横坐标
                self.x[:, 1] 的含义就是取到所有粒子的纵坐标
                参数 s 即 size 声明粒子在图中的大小
                c 即 color 声明粒子在图中的颜色
            """
            plt.scatter(self.x[:, 0], self.x[:, 1], s=300, c='r')

            """
                plt.xlim(xmin, xmax)
                设置 x 轴的显示范围
                xmin 最小值即左边的数
                xmax 最大值即右边的数
            """
            plt.xlim(self.x_bound[0], self.x_bound[1])

            """
                plt.ylim(ymin, ymax)
                设置 y 轴的显示范围
                ymin 最小值即下边的数
                ymax 最大值即上边的数
            """
            plt.ylim(self.x_bound[0], self.x_bound[1])

            """
                plt.pause(time)
                迭代停留的时间time，单位是秒
            """
            plt.pause(3)

            # 通过粒子群新的位置 self.x 更新粒子群适应度 fitness
            fitness = self.calculate_fitness(self.x)

            """
                需要更新的个体
                np.greater(list1, list2) 比较 list1 和 list2 对应位置数的大小
                list1 对应位置的数比 list2 的大，返回True，否则返回 False
                返回一个包含 True 和 False 的数组
                例如：
                    list1 = [1, 3, 2]   list2 = [2, 2, 2]
                    则返回 [False, True, False]
                update_id 保存了一列包含 True, False 的一维数组
            """

            update_id = np.greater(self.individual_best_fitness, fitness)
            # print("self.x = ", self.x)
            # print("update_id = ", update_id)
            # print("self.x[update_id] = ", self.x[update_id])
            """
                self.x[update_id]中 在 update_id 中对应位置为 True 的数会被保存下来
                例如：
                    self.x = [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]]
                    update_id = [True, False, True, True, False]
                    那么 self.x[update_id] = [[1, 1, 1], [3, 3, 3], [4, 4, 4]]
            """
            print("self.p = ", self.p)
            self.p[update_id] = self.x[update_id]
            print("self.p[update_id] = ", self.p[update_id])
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
            print("=======================================")


pso = PSO(5, 20)
pso.evolve()
plt.show()
