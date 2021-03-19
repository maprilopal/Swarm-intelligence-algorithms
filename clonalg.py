import numpy as np


class Clonalg:

    def __init__(self, **kwargs):
        self.N = kwargs.get('N', 20)
        self.b_low = kwargs.get('b_low', -10)
        self.b_up = kwargs.get('b_up', 10)
        self.generations = kwargs.get('generations', 20)
        self.c = kwargs.get('c', 0.7)
        self.beta = kwargs.get('beta', 2)
        self.p_max = kwargs.get('p_max', 3)
        self.fi = kwargs.get('fi', 0.2)
        self.k = kwargs.get('k', 0.95)
        self.if_min = kwargs.get('if_min', True)

    def make_population(self, f):
        # Generate initial population
        self.d = len(f.__code__.co_varnames) - 1
        X = (self.b_up - self.b_low) * np.random.uniform(0, 1, (self.N, self.d)) + self.b_low
        for t in range(self.generations):
            # Evaluate the fitness for each agent
            fit = np.array([f(X[i]) for i in range(self.N)])
            # Sort population by value of fitness function
            rank = np.argsort(fit)
            X = X[rank]
            fit = fit[rank]
            copies = self.__copies(X)
            # Get the best and the worst of fit
            self.best, self.worst, best_i = self.__best_and_worst(fit)
            # Get mutation range for all individual
            p = self.__p(t)
            pi = self.__p_i(p, fit)
            # Hypermutation
            hypermut = self.__hypermutation(copies, pi)
            # Check if copies are better
            stay = self.__check_copies(copies, hypermut, f)
            # Replace the rest
            a = 1 - self.c*self.N
            new_X = (self.b_up - self.b_low) * np.random.uniform(0, 1, (int(self.N*(1 - self.c)), self.d)) + self.b_low
            # Combine the remaining with new generated
            X = np.concatenate((stay, new_X), axis=0)
        fit = np.array([f(X[i]) for i in range(self.N)])
        self.best, self.worst, best_i = self.__best_and_worst(fit)
        return X[best_i]


    def __copies(self, X):
        Nc = np.array([int(self.beta*self.N/i) for i in range(1, int(self.c*self.N)+1)])
        copies = [[] for i in range(int(self.c*self.N))]
        for i in range(int(self.c*self.N)):
            copies[i] = [X[i] for j in range(Nc[i])]
        return copies

    def __hypermutation(self, copies, p_i):
        diff = self.b_up - self.b_low
        for i in range(len(copies)):
            for j in range(len(copies[i])):
                elem = copies[i][j] + p_i[i]*diff*np.random.normal(0,1)
                copies[i][j] = elem
        return copies

    def __check_copies(self, copies, hyper_copies, f):
        stay = []
        if self.if_min == True:
            for i in range(len(copies)):
                f_hyper = np.array([f(hyper_copies[i][j]) for j in range(len(copies[i]))])
                f_copy = f(copies[i][0])
                best, worst, best_i = self.__best_and_worst(f_hyper)
                if best < f_copy:
                    stay.append(hyper_copies[i][best_i])
                else:
                    stay.append(copies[i][0])
        if self.if_min == False:
            for i in range(len(copies)):
                f_hyper = np.array([f(hyper_copies[i][j]) for j in range(len(copies[i]))])
                f_copy = f(copies[i][0])
                best, worst, best_i = self.__best_and_worst(f_hyper)
                if best > f_copy:
                    stay.append(hyper_copies[i][best_i])
                else:
                    stay.append(copies[i][0])
        return np.array(stay)

    def __p(self, t):
        return self.p_max*np.exp(self.fi*t/self.generations)

    def __p_i(self, p, fit):
        #p_for_each = p*(fit - self.k*self.best)/(self.worst - self.best)
        return p*(fit - self.k*self.best)/(self.worst - self.best)

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





def f(var):
    x1, x2 = var
    return x1**2 + x2**2 - 10*(np.cos(2*np.pi*x1) + np.cos(2*np.pi*x2)) + 20

first = Clonalg()
print(first.make_population(f))
