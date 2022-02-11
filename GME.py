from collections import defaultdict
from time import time
import random
from math import copysign
import numpy as np

class Individual(dict):
    def __init__(self, method: dict, params_dict: dict):
        # self.methods = methods
        super().__init__()
        self.params_dict = params_dict

        self.method = method
        self.method.set_params(self.params_dict)
        self.method.initialize()

        self.best = np.inf
        self.best_exp = None

    def mutate(self, m_prob, mu, sigma):
        self.method.mutate(m_prob, mu, sigma)

        return self

    def evaluate(self, X_train, Y_train, X_test, Y_test, metrics):
        self.method.run_experiment(X_train, Y_train, X_test, Y_test, metrics)

        return self.method.history.get_by_idx(-1)

    def __repr__(self):
        return f"{self.method.name}: {self.params_dict}, best score = {self.best}, #best exp = {self.best_exp}"

    def __eq__(self, r_obj):
        eq = True
        for key, value in self.params_dict.items():
            eq = eq and ( r_obj.params_dict[key] == value )

        return eq and (self.method.name == r_obj.method.name)

class History(defaultdict):
    def __init__(self, stats_names = ['params', 'MAPE', 'MAE', 'train_time', 'pred_time']):
        super().__init__(list)
        self.stats_names = stats_names

    def get_by_idx(self, idx):
        res = dict()
        for key, value in self.items():
            res[key] = value[idx]

        return res


class Metric(object):
    def __init__(self, name, func):
        self.name = name
        self.func = func

    def __call__(self, y, pred):
        return self.func(y, pred)


class MetaParameters(dict):
    def __init__(self, meta_dict: dict):
        super().__init__()
        self.meta_dict = meta_dict

    def rand_init(self):

        for param_name, param_meta in self.meta_dict.items():
            if isinstance(param_meta, tuple):
                if param_meta[2] == int:
                    self[param_name] = random.randint(param_meta[0], param_meta[1])
                elif param_meta[2] == float:
                    self[param_name] = random.uniform(param_meta[0], param_meta[1])

            elif isinstance(param_meta, list):
                self[param_name] = random.choice(param_meta)

        return self

    def set_params(self, params):
        self.params = params

    def mutate(self, m_prob, mu, sigma):

        for param_name, param_meta in self.meta_dict.items():
            if random.random() < m_prob:
                if isinstance(param_meta, tuple):
                    self[param_name] += param_meta[2](param_meta[3]*random.gauss(mu, sigma))
                    if self[param_name] < param_meta[0]:
                        self[param_name] = param_meta[0]

                    if self[param_name] > param_meta[1]:
                        self[param_name] = param_meta[1]

                elif isinstance(param_meta, list):
                    self[param_name] = random.choice(param_meta)


class Method(object):
    def __init__(self, name, obj, params=None):
        self.name = name
        self.method_obj = obj

        self.params = params
        self.method = None
        self.history = History()

    def initialize(self, params=None):
        if params:
            self.params = params
        self.params.rand_init()

        # self.history.clear()
        self.method = self.method_obj(**self.params)

    def mutate(self, m_prob, mu, sigma):
        self.params.mutate(m_prob, mu, sigma)
        self.method = self.method_obj(**self.params)

    def set_params(self, params):
        if params:
            self.params = params

    def train(self, X, Y):

        self.method.fit(X, Y,)

    def predict(self, X):

        pred = self.method.predict(X)

        return pred

    def run_experiment(self, X_train, Y_train, X_test, Y_test, metrics: list, **kwargs):

        self.history['params'].append(self.params.copy())

        t_train = time()
        self.train(X_train, Y_train)
        t_train = time() - t_train

        self.history['train_time'].append(t_train)

        t_pred = time()
        pred = self.predict(X_test)
        t_pred = time() - t_pred

        self.history['pred_time'].append(t_pred)

        for metric in metrics:
            self.history[metric.name].append(metric(Y_test, pred))

    def __repr__(self):
        return f"{self.name}: {self.params}"



import random
from multiprocessing import Pool

import array
import json

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

from sklearn.ensemble import RandomForestRegressor as RFR

from utils.ga_scheme import eaMuPlusLambda

from sklearn.metrics import mean_absolute_error as mae

creator.create("FitnessMulti", base.Fitness, weights=(-1,))
creator.create("Individual", Individual, fitness=creator.FitnessMulti)

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mate(ind1, ind2):
    return ind1, ind2

def mutate(ind1, mut_prob, mu, sigma):
    return (ind1.mutate(mut_prob, mu, sigma), )

def create_individual(method, params):
    ind1 = creator.Individual(method, params.rand_init())

    return ind1

class GeneticMetaEstimator:
    def __init__(self, method, params, pop_size, iterations, norm_func=None, mut_prob=0.8, cross_prob=0.2, metrics=[Metric('MAE', mae), Metric('MAPE', mape)], input_df=True, *data):
        self.X_train, self.X_test, self.y_train, self.y_test = data

        if input_df:
            self.X_train = self.X_train.to_numpy()
            self.X_test = self.X_test.to_numpy()
            self.y_train = np.ravel(self.y_train.to_numpy())
            self.y_test = np.ravel(self.y_test.to_numpy())

        self.metrics = metrics

        self.pop_size = pop_size
        self.iterations = iterations

        self.mut_prob = mut_prob
        self.cross_prob = cross_prob

        self.norm_func = norm_func

        self.engine = base.Toolbox()

        self.engine.register("individual", create_individual, method, params)
        self.engine.register("population", tools.initRepeat, list, self.engine.individual, self.pop_size)

        self.engine.register("mate", mate)
        self.engine.register("mutate", mutate, mut_prob=mut_prob, mu=0, sigma=1)

        self.engine.register("select", tools.selTournament, tournsize=2)
        self.engine.register("evaluate", self.eval_func)

    def eval_func(self, individual):
        res = individual.evaluate(self.X_train, self.y_train, self.X_test, self.y_test, self.metrics)
        if self.norm_func:
            res = self.norm_func(res)
        res = res['pred_time'] + res['MAPE'] + res['MAE']
        individual.best = res if individual.best > res else individual.best
        if individual.best == res:
            individual.best_exp = len(individual.method.history['MAPE']) - 1

        return res,

    def get_fit_value(self, ind):
        return ind.fitness.values[0]

    def run(self, verbose=True):
        pop = self.engine.population()
        # hof = tools.HallOfFame(3, np.array_equal)
        hof = tools.HallOfFame(3)
        stats = tools.Statistics(self.get_fit_value)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

#         pop, log = eaMuPlusLambda(pop, self.engine, mu=self.pop_size, lambda_=int(self.pop_size * 0.8),
#                                   cxpb=self.cross_prob, mutpb=self.mut_prob,
#                                   ngen=self.iterations,
#                                   stats=stats, halloffame=hof, verbose=True)

        pop, log = algorithms.eaSimple(pop, self.engine, cxpb=self.cross_prob, mutpb=self.mut_prob, ngen=self.iterations,
                        stats=stats, halloffame=hof, verbose=verbose)

        print("Best = {}".format(hof[0]))
        print("Best fit = {}".format(hof[0].fitness.values[0]))
        return log, hof[0]
