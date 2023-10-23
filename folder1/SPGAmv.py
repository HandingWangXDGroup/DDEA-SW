import numpy as np
import random
from .Latin import latin_mix

# %%
class GA_mix():
    def __init__(self, lb, ub, num_contious):
        self.r = num_contious
        self.CXPB = 0.9
        self.MUTPB = 0.1
        self.NGEN = 100
        self.popsize = 100
        self.dim = len(lb)
        self.lb = lb
        self.ub = ub

    def pop_init(self):
        x = latin_mix(self.popsize, self.lb, self.ub, self.r, self.dim - self.r)
        return x

    def selectBest(self, individuals):
        index_min = np.argmin(individuals[:, -1])
        ind_best = individuals[index_min]
        return ind_best       

    def selection(self, individuals):
        tournament_size = 8
        chosen = np.empty([self.popsize, self.dim + 1])
        for i in range(self.popsize):
            select_inds = random.sample(range(self.popsize), tournament_size)
            candidates = individuals[select_inds]
            winInd = np.argmin(candidates[:, -1])
            chosen[i] = candidates[winInd]
        return chosen

    def Cross_Mutation(self, pop):
        miu = 1
        X = pop
        offspring = np.zeros([self.popsize, self.dim])
        for i in range(0, self.popsize, 2):
            newoff1 = np.zeros(self.dim)
            newoff2 = np.zeros(self.dim)
            geninfo1 = X[i, :]
            geninfo2 = X[i + 1, :]
            if random.random() <= self.CXPB:
                for j in range(self.r):
                    rand = random.random()
                    if rand <= 0.5:
                        beta = (2 * rand) ** (1.0 / (1 + miu))
                    else:
                        beta = (1 / (2 - rand * 2)) ** (1.0 / (1 + miu))
                    c1 = 0.5 * ((1 + beta) * geninfo1[j] + (1 - beta) * geninfo2[j])
                    c2 = 0.5 * ((1 - beta) * geninfo1[j] + (1 + beta) * geninfo2[j])
                    newoff1[j] = c1
                    newoff2[j] = c2
            else:
                newoff1 = geninfo1
                newoff2 = geninfo2
            offspring[i] = newoff1
            offspring[i + 1] = newoff2

        for i in range(self.popsize):
            geninfo = offspring[i, :]
            for j in range(self.r):
                rand = random.random()
                if rand <= self.MUTPB:
                    y = geninfo[j]
                    delta1 = y
                    delta2 = 1.0 - y
                    rand = random.random()
                    mut_pow = 1.0 / (miu + 1)
                    if rand <= 0.5:
                        xy = 1.0 - delta1
                        val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (miu + 1.0))
                        deltaq = val ** mut_pow - 1.0
                    else:
                        xy = 1.0 - delta2
                        val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (miu + 1.0))
                        deltaq = 1.0 - val ** mut_pow
                    y = y + deltaq
                    geninfo[j] = y
            offspring[i] = geninfo

        for i in range(0, self.popsize, 2):
            geninfo1 = offspring[i, :]
            geninfo2 = offspring[i + 1, :]

            encodelengths = len(np.binary_repr(self.ub[self.dim - self.r])) 
            encodeX = np.zeros([2, self.dim - self.r, encodelengths]) 

            for j in range(self.r, self.dim):
               b1 = np.binary_repr(int(geninfo1[j]), width = encodelengths)
               b2 = np.binary_repr(int(geninfo2[j]), width = encodelengths)

               encodeX[0][j - self.r] = np.array([int(b) for b in b1])   
               encodeX[1][j - self.r] = np.array([int(b) for b in b2])

            for j in range(self.dim - self.r):
                pos1 = random.randrange(0, encodelengths)
                pos2 = random.randrange(0, encodelengths)
                newb1 = np.empty(encodelengths)
                newb2 = np.empty(encodelengths)
                if random.random() <= self.CXPB:
                    for t in range(encodelengths):
                        if min(pos1, pos2) <= t < max(pos1, pos2):
                            newb1[t] = encodeX[1][j][t]
                            newb2[t] = encodeX[0][j][t]
                        else:
                            newb1[t] = encodeX[0][j][t]
                            newb2[t] = encodeX[1][j][t]
                else:
                    newb1 = encodeX[0][j]
                    newb2 = encodeX[1][j]

                for newb in [newb1, newb2]:
                    k = 0
                    for t in range(encodelengths):
                        if random.random() <= self.MUTPB:
                            newb[t] = random.randint(0, 1)
                    encodeX[k][j] = newb
                    k = k + 1

            power_array = np.zeros([encodelengths])
            for j in range(encodelengths):
                power_array[encodelengths - j - 1] =  2 ** j

            geninfo1[self.r - self.dim:] = np.dot(encodeX[0], power_array)
            geninfo2[self.r - self.dim:] = np.dot(encodeX[1], power_array)  
            offspring[i, :] = geninfo1
            offspring[i + 1, :] = geninfo2
        offspring = np.clip(offspring, self.lb, self.ub)
        return offspring