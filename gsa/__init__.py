import numpy as np


class Gsa:

    #Arguments:
    # f - objective function
    # N - population, number of agents
    # d - dimension of f, as we pass on one argument in function, which is list of arg, we substract 1 from number of arg
    # b_low - lower bound of search area
    # b_up - upper bound of search area
    # num_it - number of iterations
    # G - gravitational acceleration, assigned value is 1
    # eps - epsilon, small value to avoid division by zero
    def __init__(self, **kwargs):
        """
        :param args: f, N, s_low, s_up, num_it, G, eps
        """
        self.N = kwargs.get('N', 10)
        self.b_low = kwargs.get('b_low', -10)
        self.b_up = kwargs.get('b_up', 10)
        self.num_it = kwargs.get('num_it', 20)
        self.if_min = kwargs.get('if_min', True)
        self.G0 = kwargs.get('G0', 2)
        self.G = self.G0
        self.eps = kwargs.get('eps', 0.0001)
        self.kbest = kwargs.get('kbest', 0)
        self.return_all_best = kwargs.get('return_all_best', False)

    def __make_init_population(self, f):
        # Generate initial population
        X = (self.b_up - self.b_low) * np.random.random_sample((self.N, self.d)) + self.b_low
        # Evaluate the fitness for each agent
        fit = np.array([f(X[i]) for i in range(self.N)])
        best, worst, i_best = self.__best_and_worst(fit)
        # Calculate the masses from fitness
        M = self.__masses(fit, best, worst)
        # Calculate the force
        F = self.__force(M, X)
        # Calculate acceleration
        a = self.__acceleration(F, M)
        # Calculate velocity
        v = self.__velocity(np.zeros((self.N, self.d)), a)
        X = self.__positions(X, v)
        return X, M, F, a, v

    def optimize(self, f):
        self.d = len(f.__code__.co_varnames) - 1
        X, M, F, a, v = self.__make_init_population(f)
        values = []
        for i in range(self.num_it):
            fit = self.__fitness(X, f)
            self.G = self.__update_grav_force(i)
            best, worst, best_i = self.__best_and_worst(fit)
            values.append(X[best_i])
            M = self.__masses(fit, best, worst)
            Kbest = self.__Kbest(M, best_i)
            M = M[Kbest]
            X = X[Kbest]
            v = v[Kbest]
            F = self.__force(M, X)
            a = self.__acceleration(F, M)
            v = self.__velocity(v, a)
            X = self.__positions(X, v)
            if len(X) < self.kbest:
                break
        best, worst, best_i = self.__best_and_worst(self.__fitness(X, f))
        values.append(X[best_i])
        if self.return_all_best == True:
            return values
        else:
            return X[best_i]

    def __best_and_worst(self, fit):
        if self.if_min == True:
            best = np.min(fit)
            worst = np.max(fit)
            best_i = np.where(fit == best)[0][0]
        else:
            best = np.max(fit)
            worst = np.min(fit)
            best_i = np.where(fit == best)[0][0]
        return best, worst, best_i

    def __masses(self, fit, best, worst):
        m = np.array([((fit[i] - worst) / (best - worst + self.eps)) + self.eps for i in range(len(fit))])
        M = np.array([m[i] / sum(m) for i in range(len(m))])
        return M

    def __fitness(self, X, f):
        fit = np.array([f(x) for x in X])
        return fit

    def __force(self, M, X):
        F = []
        for i in range(len(M)):
            f = [0 for i in range(self.d)]
            for j in range(len(M)):
                if i != j:
                    R = np.linalg.norm(X[i] - X[j])
                    for k in range(self.d):
                        f[k] += ((self.G * M[i] * M[j])/(R**3 + self.eps)) * (X[i][k] - X[j][k])
            F.append(f)
        return np.array(F)

    def __acceleration(self, F, M):
        a = np.zeros((len(F), self.d))
        for i in range(len(F)):
            a[i] = F[i] / M[i]
        return a

    def __velocity(self, v, a):
        new_v = np.zeros((len(a), self.d))
        for i in range(len(a)):
            for j in range(self.d):
                new_v[i][j] = np.random.rand()*v[i][j] + a[i][j]
        return new_v

    def __positions(self, X, v):
        new_X = np.zeros((len(v), self.d))
        for i in range(len(v)):
            for j in range(self.d):
                new_X[i][j] = X[i][j] + v[i][j]
                if (new_X[i][j] > self.b_up) | (new_X[i][j] < self.b_low):
                    new_X[i][j] -= v[i][j]
        return new_X

    def __update_grav_force(self, t):
        beta = 20
        return self.G0*np.exp((-beta*t)/self.num_it) + self.eps
        #return self.G0*((t/self.num_it)**beta)

    def __Kbest(self, M, best_i):
        sortedM = np.argsort(M)[::-1]
        sortedM = sortedM[:-self.kbest]
        if best_i not in sortedM:
            sortedM = np.append(sortedM, best_i)
        return sortedM






