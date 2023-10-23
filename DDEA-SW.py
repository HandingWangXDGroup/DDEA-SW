import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
from matplotlib import pyplot as plt
from collections import Counter

from folder1.RBFNmv import RBFNmv
from folder1.SPGAmv import GA_mix
from folder1.Latin import latin_mix

from Benchmark import msacoprob

import warnings
warnings.filterwarnings('ignore')

def evaluate_RBFN(pop, model_list):
    if isinstance(model_list, RBFNmv):
        model_list = [model_list]
    sum_y = 0
    for model in model_list:
        temp_y = model.predict(pop)
        sum_y = sum_y + temp_y
    y = sum_y / len(model_list)
    return y[:, np.newaxis]

def candidate_solved(model):
    global lb, ub, pro
    SPGA = GA_mix(lb, ub, pro.r)
    pop  = SPGA.pop_init()
    fitness = evaluate_RBFN(pop, model)
    individuals = np.concatenate((pop, fitness), axis = 1) 
    ind_best = SPGA.selectBest(individuals)
    min_individual = ind_best[:-1]
    minimum = ind_best[-1]
    for g in range(SPGA.NGEN):
        chosen = SPGA.selection(individuals)
        nextoff = SPGA.Cross_Mutation(chosen[:, :-1])
        fit_nextoff = evaluate_RBFN(nextoff, model)
        individuals = np.concatenate((nextoff, fit_nextoff), axis = 1)
        Current_ind_best = SPGA.selectBest(individuals)
        if Current_ind_best[-1] < minimum:
            min_individual = Current_ind_best[:-1]
            minimum = Current_ind_best[-1]
    return min_individual[pro.r:], minimum       

problem_name =  ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 
                'F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F17', 'F18', 'F19', 'F20', 
                'F21', 'F22', 'F23', 'F24', 'F25', 'F26', 'F27', 'F28', 'F29', 'F30']

for problem in problem_name:
    print('-------%s-------' % problem)
    pro = msacoprob.TP1(problem)
    cub = [pro.bounds[1]] * pro.r
    clb = [pro.bounds[0]] * pro.r
    ub = cub + [pro.N_lst[0] - 1] * (pro.dim - pro.r)
    lb = clb + [0] * (pro.dim - pro.r)
    EnsembleSize = 50
    GroupSize = 10
    yb_list = []
    result_list = []
    for i in range(20):
        y_true_list = []
        ModelPool = []
        y_pred_list = []
        candidate_list = []
        EnsembleList = [] 
        path = os.getcwd()
        data_path = path + '\\Data\\' + problem + '\\Data' + str(i) + '.csv'
        x = genfromtxt(data_path, delimiter=',')
        y = pro.F(x)
        for i in range(EnsembleSize):
            model = RBFNmv(pro.dim, pro.N_lst, np.array(clb), np.array(cub), n = 4)
            model.fit(x, y)
            candidate, y_pred = candidate_solved(model)
            ModelPool.append(model)
            y_pred_list.append(y_pred)
            candidate_list.append(candidate)
        x_dis = []
        candidate_array = np.array(candidate_list)
        for i in range(pro.dim - pro.r):
            count = Counter(candidate_array[:, i])
            most_common = count.most_common(1)
            x_dis.append(int(most_common[0][0]))
        predict_value_index = np.argsort(y_pred_list)
        GroupIndex = predict_value_index.reshape(GroupSize, int(EnsembleSize/GroupSize))
        for i in range(GroupSize):
            model_index = int(np.random.choice(GroupIndex[i], 1))
            EnsembleList.append(ModelPool[model_index])        
        SPGA = GA_mix(lb, ub, pro.r)
        pop  = SPGA.pop_init()
        pop[:, pro.r:] = x_dis
        fitness = evaluate_RBFN(pop, EnsembleList)
        individuals = np.concatenate((pop, fitness), axis = 1) 
        ind_best = SPGA.selectBest(individuals)
        min_individual = ind_best[:-1]
        minimum = ind_best[-1]
        for g in range(SPGA.NGEN):
            chosen = SPGA.selection(individuals)
            nextoff = SPGA.Cross_Mutation(chosen[:, :-1])
            nextoff[:, pro.r:] = x_dis
            fit_nextoff = evaluate_RBFN(nextoff, EnsembleList)
            individuals = np.concatenate((nextoff, fit_nextoff), axis = 1)
            Current_ind_best = SPGA.selectBest(individuals)
            if Current_ind_best[-1] < minimum:
                min_individual = Current_ind_best[:-1]
                minimum = Current_ind_best[-1]
            y_true = pro.F(min_individual.tolist())
            y_true_list.append(y_true)
        yb = y_true_list[-1]
        yb_list.append(yb)
        result_list.append(y_true_list)

    mean = np.mean(yb_list)
    std = np.std(yb_list)
    print('mean: %.2f' % mean)
    print('std: %.2f' % std)