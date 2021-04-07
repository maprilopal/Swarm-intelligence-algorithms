import numpy as np

class Imperialistic:
    def __init__(self, **kwargs):
        self.N = kwargs.get('N', 20)
        self.b_low = kwargs.get('b_low', -10)
        self.b_up = kwargs.get('b_up', 10)
        self.num_it = kwargs.get('num_it', 20)
        self.if_fit = kwargs.get('if_fit', False)
        self.if_min = kwargs.get('if_min', True)
        self.beta = kwargs.get('beta', 2)
        self.gamma = kwargs.get('gamma', 0.25)
        self.numImp = kwargs.get('numberImp', int(0.4*self.N))
        self.numCol = int(self.N - self.numImp)


    def optimize(self, f):
        self.d = len(f.__code__.co_varnames) - 1
        X = (self.b_up - self.b_low) * np.random.rand(self.N, self.d) + self.b_low
        costCountry = np.array([f(X[i]) for i in range(self.N)])
        sortedByCost = np.argsort(costCountry)[::-1]
        X = X[sortedByCost]
        costCountry = costCountry[sortedByCost]
        normCost = costCountry[:self.numImp] - np.max(costCountry[:self.numImp])
        sumImp = np.sum(normCost[:self.numImp])
        powerImp = np.abs(normCost/sumImp)
        numEmp = np.round(self.numCol*powerImp)

        return numEmp


def Matyas(var):
    x1, x2 = var
    return 0.26*(x1**2 + x2**2) - 0.48*(x1*x2)

alg = Imperialistic()
print(alg.optimize(Matyas))