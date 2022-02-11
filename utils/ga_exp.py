from deap import tools, base
from multiprocessing import Pool
from ga_scheme import eaMuPlusLambda
# from deap.algorithms import eaMuPlusLambda
from numpy import random as rnd
import numpy as np
from deap import creator
from deap import benchmarks

creator.create("BaseFitness", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.BaseFitness)


def mutation(individual):
    n = len(individual)
    for i in range(n):
        if rnd.random() < n * 0.15:
            individual[i] += rnd.normal(0.0, 0.2)
            individual[i] = np.clip(individual[i], -5, 5)
    return individual,


class SimpleGAExperiment:
    def factory(self):
        return rnd.random(self.dimension) * 10 - 5

    def __init__(self, function, dimension, pop_size, iterations):
        self.pop_size = pop_size
        self.iterations = iterations
        self.mut_prob = 0.8
        self.cross_prob = 0.2

        self.function = function
        self.dimension = dimension

        # mut gauss 0 0.5 0.1, tour 8, TwoPoint = 8.35
        # mut gauss 0 0.75 0.1, tour 2, TwoPoint = 9.94
        # mut gauss 0 0.75 0.1, roulet, TwoPoint = 9.59
        # mut gauss 0 0.75 0.1 selBest, TwoPoint = 9.69
        # mut gauss 0 0.75 0.1 tour 2 Uniform 0.15 = 9.92
        # mut gauss 0 1 0.1 tour 5 Uniform 0.25 = 8.94
        # pop 50 mut gauss 0 0.75 0.1, tour 2, TwoPoint = 9.69

        self.pool = Pool(10)
        self.engine = base.Toolbox()
        self.engine.register("map", self.pool.map)
        # self.engine.register("map", map)
        self.engine.register("individual", tools.initIterate, creator.Individual, self.factory)
        self.engine.register("population", tools.initRepeat, list, self.engine.individual, self.pop_size)
        self.engine.register("mate", tools.cxTwoPoint)
        self.engine.register("mutate", tools.mutGaussian, mu=0, sigma=0.75, indpb=0.1)
        # self.engine.register("mutate", mutation)
        self.engine.register("select", tools.selTournament, tournsize=2)
        # self.engine.register("select", tools.selRoulette)
        # self.engine.register("select", tools.selBest)
        self.engine.register("evaluate", self.function)

    def run(self):
        pop = self.engine.population()
        hof = tools.HallOfFame(3, np.array_equal)
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        pop, log = eaMuPlusLambda(pop, self.engine, mu=self.pop_size, lambda_=int(self.pop_size * 0.8),
                                  cxpb=self.cross_prob, mutpb=self.mut_prob,
                                  ngen=self.iterations,
                                  stats=stats, halloffame=hof, verbose=True)
        print("Best = {}".format(hof[0]))
        print("Best fit = {}".format(hof[0].fitness.values[0]))
        return log


from functions import rastrigin

if __name__ == "__main__":
    def function(x):
        res = rastrigin(x)
        return res,


    dimension = 100
    pop_size = 100
    iterations = 10000
    scenario = SimpleGAExperiment(function, dimension, pop_size, iterations)
    log = scenario.run()
    from draw_log import draw_log

    draw_log(log)
