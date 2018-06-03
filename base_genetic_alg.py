#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np


MAX_GENERATION = 200
POPULATION_SIZE = 2000
DNA_SIZE = 10


class GeneticAlg(object):

    def __init__(self, dna_size=10, cross_rate=0.8, mutation_rate=0.003):
        """
        :param dna_size:
        :param cross_rate: mating probability (DNA crossover)
        :param mutation_rate:
        """
        self.dna_size = dna_size
        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate

    def translate_dna(self, population):
        """
        将DNA信息转化为行动方式
        :param population: 种群中所有的DNA[种群大小, DNA大小]
        :return: 返回种群中所有的行为模式。 [种群大小, x坐标0-10]
        """
        # return np.sum(population, axis=1)
        return population.dot(
            2 ** np.arange(self.dna_size)[::-1]
        ) / float(2 ** self.dna_size - 1) * 100

    @staticmethod
    def enviroment_state(features):
        """
        根据特征， 计算环境给他的分数
        :param features: 种群中所有个体的特征， [种群大小, 特征个数]
        :return: [种群大小, 分数，y轴值]
        """
        x = features.reshape((-1, ))
        return 10 * (np.sin(x) + np.cos(x))

    @staticmethod
    def get_fitness(states):
        """
        根据生活状态计算种群所有个体的适应程度
        :param states: 种群所有个体的生存状态
        :return: 种群所有个体的适应度
        """
        return states - np.min(states) + 1e-3

    def select(self, population, fitness):
        """
        物竞天择， 适者生存
        根据种群所有个体的适应度，淘汰弱势群体
        为了方便， 我们返回的群体长度保持不变， 只是根据概率， 适应度高的副本保留多点
        :param population: 种群
        :param fitness: 种群所有个体的适应度
        :return: 存货下来的个体
        """
        population_size = len(population)
        probs = fitness / fitness.sum()
        idx = np.random.choice(np.array(population_size), size=population_size, p=probs)
        return population[idx]

    def gen_child(self, parent, population):
        """
        每一个parent 在种群中选择配偶交配
        1） 先选择配偶交配DNA
        2） DNA一定概率发生突变
        :param parent:
        :param population:
        :return: 孩子
        """
        population_size = len(population)
        child = None
        if np.random.rand() < self.cross_rate:
            # 选择配偶
            mate = np.random.randint(0, population_size, size=1)
            cross_points = np.random.randint(0, 2, size=self.dna_size).astype(np.bool)
            child = parent
            # 交配时， 部分基因由另一半继承
            child[cross_points] = population[mate, cross_points]
            # 基因突变
            for point in range(self.dna_size):
                if np.random.rand() < self.mutation_rate:
                    child[point] = 1 if child[point] == 0 else 0
        else:
            # 有一定概率不选择配偶
            child = parent
        return child


def main():
    alg = GeneticAlg(dna_size=DNA_SIZE)
    population = np.random.randint(2, size=(POPULATION_SIZE, DNA_SIZE))
    for i in range(MAX_GENERATION):
        features = alg.translate_dna(population)
        states = alg.enviroment_state(features)
        fitness = alg.get_fitness(states)

        print("genetic", i,
              "best count", np.sum(np.round(states, 0) == np.round(np.max(states), 0)),
              "avg state", round(np.mean(states), 2),
              "best feature", round(features[np.argmax(fitness)]),
              "best dna", population[np.argmax(fitness), :]
              )

        population = alg.select(population, fitness)
        pre_population = population.copy()
        # 开始产生下一代种群
        for parent in population:
            # 繁衍下一代
            child = alg.gen_child(parent, pre_population)
            # 用新的一代替换上一代
            parent[:] = child


if __name__ == '__main__':
    main()
