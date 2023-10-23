import numpy as np
from scipy import stats
from .SPGAmv import GA_mix
from .Latin import latin_mix
import gower

# %%
class RBFNmv():
    def __init__(self, dim, N_lst, cxmin, cxmax, n):
        '''
        :param dim: the dimension of decision variables
        :param N_lst: the number of values for discrete variables
        :param cxmin: the lower bounds of continuous decision variables
        :param cxmax: the upper bounds of continuous decision variables
        '''
        self.dim = dim
        self.num_neurons = None
        self.sigma = None
        self.centers = None
        self.cxmin = cxmin
        self.cxmax = cxmax

        self.weights = None
        self.bias = None

        self.n = n

        self.N_lst = N_lst
        self.o = len(N_lst) # number of discrete variables
        self.r = self.dim - self.o # number of continuous variables
    
    def _cal_beta(self, X, n):
        cat_features = [False] * self.r + [True] * self.o
        dis_matrix = gower.gower_matrix(X, cat_features = cat_features)
        Dmax = np.max(dis_matrix)
        beta1 = Dmax * (self.dim * X.shape[0]) ** (-1 / self.dim)
        self.beta = beta1 * 2 ** (4*(n-1))
        return self.beta


    # Gower distance based Gaussian kernel
    def kernel_(self, data_point):
        n1 = data_point.shape[0]
        n2 = self.centers.shape[0]
        D = data_point.shape[1]

        rdistMat = np.zeros((n1, n2))
        xr1 = data_point[:, :self.r]
        xr2 = self.centers[:, :self.r]
        for i in range(n1):
            rdistMat[i, :] = np.sum(np.abs(xr1[i] - xr2)/(self.cxmax - self.cxmin), axis=1)

        cdisMat = np.zeros((n1, n2))
        for i in range(n1):
            cdisMat[i, :] = np.sum(data_point[i, self.r:] != self.centers[:, self.r:], axis=1)
        distMat = (rdistMat + cdisMat) / D

        return np.exp(- distMat / self.beta)

    def kmeans(self, X, n_clusters):

        nums, dim = X.shape

        c_s = np.random.choice(np.arange(nums), n_clusters, replace=False)
        centers = X[c_s, :]

        clusters_lst = [[] for i in range(n_clusters)]

        delta = np.inf
        c = 0
        while (delta > 1e-4 and c < 500):
            c_f = centers.copy()

            for i in range(nums):
                dist_r = np.sum(np.abs(X[i, :self.r] - centers[:, :self.r])/(self.cxmax - self.cxmin),
                                axis=1)
                t = X[i, self.r:] - centers[:, self.r:]
                t[t != 0] = 1
                dist_c = np.sum(t, axis=1)
                dist = dist_c + dist_r
                ind = np.argmin(dist)
                clusters_lst[ind].append(i)

            for k in range(n_clusters):
                centers[k, :self.r] = np.mean(X[clusters_lst[k], :self.r], axis=0)
                t = stats.mode(X[clusters_lst[k], self.r:])[0]
                centers[k, self.r:] = t[0] if len(t != 0) else centers[np.random.randint(0, n_clusters), self.r:]

            clusters_lst = [[] for i in range(n_clusters)]
            c_b = centers[:]
            delta = np.sum((c_f - c_b) ** 2)
            c += 1

        return c_f

    def pinv(self, A, reg):
        return np.linalg.inv(reg * np.eye(A.shape[1]) + A.T.dot(A)).dot(A.T)

    def fit(self, X, Y):

        self.num_neurons = int(X.shape[0] / 2)
        self.centers = self.kmeans(X, n_clusters=self.num_neurons)

        self._cal_beta(X, self.n)
        G = self.kernel_(X)
        temp = np.column_stack((G, np.ones((X.shape[0]))))
        temp = np.dot(np.linalg.pinv(temp), Y)
        self.weights = temp[:self.num_neurons]
        self.bias = temp[self.num_neurons]

    def predict(self, X):
        G = self.kernel_(X)
        predictions = np.dot(G, self.weights) + self.bias
        return predictions
    
    def score(self, X, y):
        MSE = np.sum((y - self.predict(X)) ** 2) / len(y)
        R2 = 1 - MSE / np.var(y)
        return R2
    
    def rmse(self, X, y):
        RMSE = np.sqrt(np.sum((y - self.predict(X)) ** 2) / len(y))
        return RMSE      