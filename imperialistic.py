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
        self.xi = kwargs.get('xi', 0.1)
        self.numImp = kwargs.get('numberImp', int(0.4*self.N))
        self.numCol = int(self.N - self.numImp)


    def optimize(self, f):
        self.d = len(f.__code__.co_varnames) - 1
        # Initialize countries
        X = (self.b_up - self.b_low) * np.random.rand(self.N, self.d) + self.b_low
        # Cost of every country
        costCountry = np.array([f(X[i]) for i in range(self.N)])
        sortedByCost = np.argsort(costCountry)
        X = X[sortedByCost]
        costCountry = costCountry[sortedByCost]
        imperiors = X[:self.numImp]
        colonies = X[self.numImp:]
        # Normalized cost
        normCost = costCountry[:self.numImp] - np.max(costCountry[:self.numImp])
        # Power of every imperialist
        sumImp = np.sum(normCost[:self.numImp])
        powerImp = np.abs(normCost/sumImp)
        # Initial number of colonies of an empire
        numColOfImp = np.round(self.numCol*powerImp)
        # Divide colonies in random way
        colOfImp = self.__divideColonies(colonies, numColOfImp)
        # Total power of empire
        totalPowerEmp = self.__totalPowerOfEmpire(costCountry, colOfImp)




        return totalPowerEmp


    def __divideColonies(self, colonies, numColOfImp):
        divColonies = []
        sumCol = 0
        if sum(numColOfImp) != self.numCol:
            if sum(numColOfImp) > self.numCol:
                numColOfImp[0] -= sum(numColOfImp) - self.numCol
            else:
                numColOfImp[0] += self.numCol - sum(numColOfImp)
        for num in numColOfImp:
            divColonies.append(colonies[sumCol:sumCol+int(num)])
            sumCol += int(num)
        return divColonies

    def __totalPowerOfEmpire(self, costOfCountry, coloniesOfImp):
        totalPowerEmp = []
        for i in range(self.numImp):
            if coloniesOfImp[i].size == 0:
                meanColony = 0
            else:
                meanColony = np.mean(coloniesOfImp[i])
            totalPowerEmp.append(costOfCountry[i]+self.xi*meanColony)
        return totalPowerEmp






def Matyas(var):
    x1, x2 = var
    return 0.26*(x1**2 + x2**2) - 0.48*(x1*x2)

alg = Imperialistic()
print(alg.optimize(Matyas))