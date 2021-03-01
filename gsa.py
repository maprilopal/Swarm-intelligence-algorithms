import numpy as np
from graphics import show_graphics
import time


class Gsa:

    #Arguments:
    # f - objective function
    # N - population, number of agents
    # d - dimension of f, as we pass on one argument in function, which is list of arg, we substract 1 from number of arg
    # s_low - lower border of search area
    # s_up - upper border of search area
    # num_it - number of iterations
    # G - gravitational acceleration, assigned value is 1
    # eps - epsilon, small value to avoid division by zero
    # fit - fitness function, OPTIONAL, assigned value is objective function
    # if_fit - if fitness function is assigned
    def __init__(self, **kwargs):
        """
        :param args: f, N, s_low, s_up, num_it, G, eps, fit
        """
        self.N = kwargs.get('N', 10)
        self.b_low = kwargs.get('b_low', -10)
        self.b_up = kwargs.get('b_up', 10)
        self.num_it = kwargs.get('num_it', 20)
        self.if_fit = kwargs.get('if_fit', False)
        self.if_min = kwargs.get('if_min', True)
        self.G0 = kwargs.get('G0', 2)
        self.G = self.G0
        self.eps = kwargs.get('eps', 0.0001)
        self.kbest = kwargs.get('kbest', 1)
        self.return_all_best = kwargs.get('return_all_best', False)

    def change_G(self, new_G):
        self.G = new_G

    def change_eps(self, new_eps):
        self.eps = new_eps

    def change_fitF(self, new_fit):
        self.fit = new_fit
        self.if_fit = True

    def __make_init_population(self, f):
        # Generate initial population
        X = (self.b_up - self.b_low) * np.random.random_sample((self.N, self.d)) + self.b_low
        #X = np.array([[2,2],[2,2],[5,2],[4,5],[1,3]])
        #X = np.random.randint(-5, 5, (self.N, self.d))
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

    def __make_init_populationFit(self, f, fitF):
        # Generate initial population
        X = (self.b_up - self.b_low) * np.random.rand(self.N, self.d) + self.b_low
        # Evaluate the fitness for each agent
        values_f = np.array([f(X[i]) for i in range(self.N)])
        fit = np.array([fitF(values_f[i]) for i in range(self.N)])
        best, worst, i_best = self.__best_and_worst(fit)
        # Calculate the masses from fitness
        M = self.__masses(fit, best, worst)
        # Calculate the force
        F = self.__force(M, X, M.argsort())
        # Calculate acceleration
        a = self.__acceleration(F, M)
        # Calculate velocity
        v = self.__velocity(np.zeros((self.N, self.d)), a)
        #Calculate new positions
        X = self.__positions(X, v)
        return X, M, F, a, v

    def optimize(self, f):
        self.d = len(f.__code__.co_varnames) - 1
        X, M, F, a, v = self.__make_init_population(f)
        #print("X\n", X)
        values = []
        for i in range(self.num_it):
            #print("************* ", i, " ******************")
            fit = self.__fitness(X, f)
            #print("fit\n", fit)
            self.G = self.__update_grav_force(i)
            #print("G: ", self.G)
            best, worst, best_i = self.__best_and_worst(fit)
            values.append(X[best_i])
            #print(i, ": ", X[best_i],"--->", best, "                               worst: ", worst)
            #print("best: ", best)
            #print("worst: ", worst)
            #print("best_i ", best_i)
            M = self.__masses(fit, best, worst)
            #print("M\n", M)
            Kbest = self.__Kbest(M, best_i)
            # Get rid of worsts agents
            M = M[Kbest]
            X = X[Kbest]
            v = v[Kbest]
            #calculate force
            F = self.__force(M, X)
            #print("F\n",F)
            a = self.__acceleration(F, M)
            #print("a\n",a)
            v = self.__velocity(v, a)
            #print("v\n",v)
            X = self.__positions(X, v)
            if len(X) == 1:
                break
        #values = []
        #for x in X:
            #values.append(f(x))
        best, worst, best_i = self.__best_and_worst(self.__fitness(X, f))
        values.append(X[best_i])
        if self.return_all_best == True:
            return values
        else:
            return X[best_i]

    def optimizeFit(self, f, fitF):
        self.if_fit = True
        self.d = len(f.__code__.co_varnames) - 1
        X, M, F, a, v = self.__make_init_populationFit(f, fitF)
        X = self.__positions(X,v)
        for i in range(self.num_it):
            fit = self.__fitnessFit(X, f, fitF)
            self.G = self.__update_grav_force(i)
            best, worst, best_i = self.__best_and_worst(fit)
            M = self.__masses(fit, best, worst)
            Kbest = self.__Kbest(M, best_i)
            M = M[Kbest]
            X = X[Kbest]
            v = v[Kbest]
            F = self.__force(M, X)
            a = self.__acceleration(F, M)
            v = self.__velocity(v, a)
            if len(X) == 1:
                break
        values = []
        for x in X:
            values.append(f(x))
        best, worst, best_i = self.__best_and_worst(values)
        return X[best_i][0]

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

    def __fitnessFit(self, X, f, fitF):
        values_f = np.array([f(x) for x in X])
        fit = np.array([fitF(v) for v in values_f])
        return fit

    def __force(self, M, X):
        #F = np.zeros((len(Kbest), self.d))
        F = []
        for i in range(len(M)):
            f = [0 for i in range(self.d)]
            #print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% i: ",i)
            for j in range(len(M)):
                if i != j:
                    #print("j: ", j)
                    R = np.linalg.norm(X[i] - X[j])
                    for k in range(self.d):
                        f[k] += ((self.G * M[i] * M[j])/(R**3 + self.eps)) * (X[i][k] - X[j][k])
                        #F[i][k] += ((self.G * M[i] * M[j])/(R + self.eps)) * (X[i][k] - X[j][k]) * np.random.rand()
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
                r = np.random.rand()
                a1 = v[i][j]
                a2 = a[i][j]
                #new_v[i][j] = np.random.rand()*v[i][j] + a[i][j]
                new_v[i][j] = r*a1 + a2
        return new_v

    def __positions(self, X, v):
        new_X = np.zeros((len(v), self.d))
        for i in range(len(v)):
            for j in range(self.d):
                new_X[i][j] = X[i][j] + v[i][j]
                if (new_X[i][j] > self.b_up) | (new_X[i][j]< self.b_low):
                    new_X[i][j] -= v[i][j]
                #if new_X[i][j] > self.b_up:
                    #new_X[i][j] = self.b_up - np.random.rand()
                #if new_X[i][j] < self.b_low:
                    #new_X[i][j] = self.b_low + np.random.rand()
        return new_X

    def __update_grav_force(self, t):
        #beta = np.random.rand()
        beta = 20
        return self.G0*np.exp((-beta*t)/self.num_it) + self.eps

    def __Kbest(self, M, best_i):
        sortedM = np.argsort(M)[::-1]
        sortedM = sortedM[:-self.kbest]
        if best_i not in sortedM:
            sortedM = np.append(sortedM, best_i)
        return sortedM



#def f(var):
#    x1, x2 = var
#    return 4*(x1**2) - 2.1*(x1**4) + (x1**6)/3 + x1*x2 -4*(x2**2) +4*(x2**4)

def f(var):
    x1, x2 = var
    return x1**2 + x2**2 - 10*(np.cos(2*np.pi*x1) + np.cos(2*np.pi*x2)) + 20

def f2(var):
    x1, x2 = var
    return x1**3 + x2**3

def f3(var):
    x1, x2 = var
    return np.sin(x1+x2**2)

def ffit(var):
    x1 = var
    return 1/x1

def fitExp4(var):
    x = var
    return 1+np.exp(x)

def Matyas(var):
    x1, x2 = var
    return 0.26*(x1**2 + x2**2) - 0.48*(x1*x2)

#ar = {'N': 5, 'b_low': -5.12, 'b_up': 5.12, 'num_it': 10}
values = []
start = time.time()
for i in range(1):
    example = gsa(N = 100, b_low= -10, b_up= 10, num_it= 100, G = 100)
    res = example.optimize(Matyas)
    values.append(res)
    #print(res)
    #print(res, "--->", f2(res))
#print(f([0,0]))
end = time.time()
print("time: ", end - start)
print(np.mean(values, axis=0))
#example_all = gsa(N=50, b_low=-5.12, b_up=5.12, num_it=50, G=100, return_all_best=True)
#print(example_all.optimize(f2))

#graphsMatyas = show_graphics()
#graphsMatyas.plot_f_3D_plotly(Matyas)






