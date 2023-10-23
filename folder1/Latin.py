import numpy as np
def latin_mix(N, lower_bound, upper_bound, num_continuous, num_discret):
    '''
    N - The size of the sample data
    lower_bound - Lower bound of the decision variable
    upper_bound - Upper bound of the decision variable
    num_continuous - number of contoinuous variable
    num_discret - number pf discret variable
    '''
    lb = lower_bound
    ub = upper_bound
    D = num_continuous + num_discret 
    xmax = np.array(ub)
    xmin = np.array(lb)
    xmax[-num_discret:] = xmax[-num_discret:] + 1
    area = xmax - xmin 
    d = 1.0 / N 
    result = np.empty([N, D])
    temp = np.empty([N])
    for i in range(D): 
        for j in range(N): 
            temp[j] = np.random.uniform(low=j * d, high=(j + 1) * d, size=1)[0]
        np.random.shuffle(temp) 
        for j in range(N):
            result[j, i] = temp[j]
        result[:, i] = result[:, i] * area[i] + xmin[i]
    for i in range(num_continuous, D):
        result[:, i] = np.fix(result[:, i])
    return result