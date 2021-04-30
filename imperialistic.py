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
        d = len(f.__code__.co_varnames) - 1
        # Initialize countries
        X = (self.b_up - self.b_low) * np.random.rand(self.N, d) + self.b_low
        # Cost of every country
        costCountry = np.array([f(X[i]) for i in range(self.N)])
        sortedByCost = np.argsort(costCountry)
        X = X[sortedByCost]
        costCountry = costCountry[sortedByCost]
        imperiors = X[:self.numImp]
        countries= X[self.numImp:]
        # Normalized cost
        normCost = costCountry[:self.numImp] - np.max(costCountry[:self.numImp])
        # Power of every imperialist
        sumImp = np.sum(normCost[:self.numImp])
        powerImp = np.abs(normCost/sumImp)
        # Initial number of colonies of an empire
        numColOfImp = np.round(self.numCol*powerImp)
        # Divide colonies in random way
        colonies = self.__divideColonies(countries, numColOfImp)
        # Move the colonies toward their revelant imperialst
        colonies = self.__moveColonies(colonies, imperiors, d)
        # Check if the imperialist have better positions than a colony
        imperiors, colonies = self.__checkPosition(colonies, imperiors, f)

        # Total cost of an empire (total power)
        totalCostEmp = self.__totalPowerOfEmpire(costCountry, colonies)
        # Normalized total cost
        normTotalCostEmp = totalCostEmp - np.max(totalCostEmp)
        # Possession probability of each empire
        P = np.abs(normTotalCostEmp/ np.sum(normTotalCostEmp))
        R = np.random.uniform(0,1, size = np.size(P))
        D = P - R
        maxImp = np.argmax(D)

        # Imperialistic Competition







        return newColonies


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

    def __moveColonies(self, colonies, imperiors, d):
        for imp in range(len(imperiors)):
            for col in range(len(colonies[imp])):
                for i in range(d):
                    valCol = colonies[col][i]
                    valImp = imperiors[imp][i]
                    diff = imperiors[imp][i] - colonies[imp][col][i]
                    if diff >= 0:
                        colonies[col][i] += np.random.uniform(0, self.beta*diff)
                    else:
                        colonies[col][i] += np.random.uniform(self.beta*diff, 0)
        return colonies

    def __checkPosition(self, colonies, imperiors, f):
        for i in range(len(imperiors)):
            imperior = imperiors[i]
            for colony in colonies[i]:
                if self.if_min == True:
                    if f(imperior) < f(colony):
                        imperiors[i] = colonies[i]
                        colonies[i] = imperior
                        imperior = colonies[i]
                if self.if_min == False:
                    if f(imperior) > f(colony):
                        imperiors[i] = colonies[i]
                        colonies[i] = imperior
                        imperior = colonies[i]
        return imperiors, colonies

    def __totalPowerOfEmpire(self, costOfCountry, coloniesOfImp):
        totalPowerEmp = []
        for i in range(self.numImp):
            if coloniesOfImp[i].size == 0:
                meanColony = 0
            else:
                meanColony = np.mean(coloniesOfImp[i])
            totalPowerEmp.append(costOfCountry[i]+self.xi*meanColony)
        return totalPowerEmp

    #def __peekWeakestColony(self, colonies, f):
        #weakest = colonies[len(colonies) - 1][0]
        #for i in range(1, len(colonies(len(colonies) - 1))):
            #if self.if_min == True:
                #if f(weakest) < f(colonies[i]):
                    #weakest =







def Matyas(var):
    x1, x2 = var
    return 0.26*(x1**2 + x2**2) - 0.48*(x1*x2)

alg = Imperialistic()
print(alg.optimize(Matyas))