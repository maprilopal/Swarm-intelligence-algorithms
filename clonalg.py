import numpy as np
from graphics import Graphics
import pandas as pd

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

    def optimize(self, f):
        # Generate initial population
        self.d = len(f.__code__.co_varnames) - 1
        X = (self.b_up - self.b_low) * np.random.uniform(0, 1, (self.N, self.d)) + self.b_low
        progress = []
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
            progress.append(X[best_i])
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
        progress.append(X[best_i])
        return [X[best_i], progress]

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


class StatisticClonalg(Clonalg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.repeat = kwargs.get('repeat', 10)

    def check_N(self, f):
        result = []
        values = []
        for N in self.N:
            clonalg = Clonalg(N=N)
            for i in range(self.repeat):
                best = clonalg.optimize(f)[0]
                values.append(best)
            mean = np.mean(values, axis=0)
            result.append((N, mean, f(mean)))
        return pd.DataFrame(result, columns=['N', 'best', 'f(best)'])

    def check_gen(self, f):
        result = []
        for gen in self.generations:
            clonalg = Clonalg(generations=gen)
            result.append((gen, clonalg.optimize(f)[0]))
        return result

    def check_p(self, f):
        result = []
        for p in self.p_max:
            clonalg = Clonalg(p_max=p)
            result.append((p, clonalg.optimize(f)[0]))
        return result

    def check_beta(self, f):
        result = []
        for beta in self.beta:
            clonalg = Clonalg(beta=beta)
            result.append((beta, clonalg.optimize(f)[0]))
        return result

    def check_fi(self, f):
        result = []
        for fi in self.fi:
            clonalg = Clonalg(fi=fi)
            result.append((fi, clonalg.optimize(f, fi=fi)))
        return result

    def check_c(self, f):
        result = []
        for c in self.c:
            clonalg = Clonalg(c=c)
            result.append((c, clonalg.optimize(f)))
        return result

    def check_k(self, f):
        result = []
        for k in self.k:
            clonalg = Clonalg(k=k)
            result.append((k, clonalg.optimize(f)))
        return result

    def check_all(self, f):
        result = []
        values = []
        for N in self.N:
            for gen in self.generations:
                for p in self.p_max:
                    for beta in self.beta:
                        for fi in self.fi:
                            for c in self.c:
                                for k in self.k:
                                    clonalg = Clonalg(N=N, generations=gen, p=p, beta=beta, fi=fi, c=c, k=k)
                                    for i in range(self.repeat):
                                        best = clonalg.optimize(f)[0]
                                        values.append(best)
                                    mean = np.mean(values, axis=0)
                                    result.append((N, gen, p, beta, fi, c, k, mean, f(mean)))
        return pd.DataFrame(result, columns=['N', 'gen', 'p','beta','fi','c','k', 'best', 'f(best)'])







def f(var):
    x1, x2 = var
    return x1**2 + x2**2 - 10*(np.cos(2*np.pi*x1) + np.cos(2*np.pi*x2)) + 20
N = list(range(10,20,10))
gen = list(range(5,20,5))
p = list(map(lambda x: x/10.0, range(5,35,10)))
beta = list(map(lambda x: x/10.0, range(5,35,10)))
fi = list(map(lambda x: x/10.0, range(1,8,2)))
c = list(map(lambda x: x/10.0, range(1,6,1)))
k = list(map(lambda x: x/10.0, range(1,8,2)))
ex = StatisticClonalg(N=N, generations=gen, p_max=p, beta=beta, fi=fi, c=c, k=k)
data = ex.check_all(f)
#df = pd.DataFrame(data, columns=['N', 'value'])
print(data)


