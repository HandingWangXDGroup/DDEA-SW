import sys
import os
sys.path.append('.')
# sys.path.append('./folder1')
# sys.path.append('./Benchmark')
from folder1.Latin import latin_mix
from Benchmark import msacoprob
import numpy as np

problem_list = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 
                'F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F17', 'F18', 'F19', 'F20', 
                'F21', 'F22', 'F23', 'F24', 'F25', 'F26', 'F27', 'F28', 'F29', 'F30']
for problem in problem_list:
    path = problem
    os.mkdir(path)
    pro = msacoprob.TP1(problem)
    cub = [pro.bounds[1]] * pro.r
    clb = [pro.bounds[0]] * pro.r
    ub = cub + [pro.N_lst[0] - 1] * (pro.dim - pro.r)
    lb = clb + [0] * (pro.dim - pro.r)
    for i in range(1):
        x = latin_mix(100, lb, ub, pro.r, pro.dim - pro.r)
        np.savetxt(problem + '//Data' + str(i) + '.csv', x, delimiter=',')