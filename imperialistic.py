import copy

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
        imperialists = X[:self.numImp]
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
        # Elimiate powerless empires
        a = self.__eliminatePowerlessEmpires(colonies, imperialists)


        # Move the colonies toward their revelant imperialst
        colonies = self.__moveColonies(colonies, imperialists, d)
        # Check if the imperialist have better positions than a colony
        imperiors, colonies = self.__checkPosition(colonies, imperialists, f)

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
        colonies, imperialists = self.__competition(colonies, imperialists, maxImp, f)
        i = imperialists
        print(i)








        return colonies


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

    def __moveColonies(self, colonies, imperialists, d):
        for imp in range(len(imperialists)):
            for col in range(len(colonies[imp])):
                for i in range(d):
                    valCol = colonies[col][i]
                    valImp = imperialists[imp][i]
                    diff = imperialists[imp][i] - colonies[imp][col][i]
                    if diff >= 0:
                        colonies[col][i] += np.random.uniform(0, self.beta*diff)
                    else:
                        colonies[col][i] += np.random.uniform(self.beta*diff, 0)
        return colonies

    def __checkPosition(self, colonies, imperialists, f):
        for i in range(len(imperialists)):
            imperialist = copy.copy(imperialists[i])
            for j in range(len(colonies[i])):
                if self.if_min == True:
                    if f(imperialists[i]) < f(colonies[i][j]):
                        imperialists[i] = colonies[i][j]
                        colonies[i][j] = imperialist
                if self.if_min == False:
                    if f(imperialists[i]) > f(colonies[i][j]):
                        imperialists[i] = colonies[i][j]
                        colonies[i] = imperialist
        return imperialists, colonies

    def __eliminatePowerlessEmpires(self, colonies, imperiors):
        powerlessImperiors = np.argwhere(len(colonies) == 0)
        return powerlessImperiors


    def __totalPowerOfEmpire(self, costOfCountry, colonies):
        totalPowerEmp = []
        for i in range(len(colonies)):
            if colonies[i].size == 0:
                meanColony = 0
            else:
                meanColony = np.mean(colonies[i])
            totalPowerEmp.append(costOfCountry[i]+self.xi*meanColony)
        return totalPowerEmp

    def __competition(self, colonies, imperialists, bestImperialist, f):
        if not colonies[-1]:
            weakest = colonies[-1][0]
            for i in range(1, len(colonies[-1])):
                if self.if_min == True:
                    if f(weakest) < f(colonies[i]):
                        weakest = colonies[i]
        else:
            colonies = colonies[:-1]
        imperialists[bestImperialist].append(weakest)
        return colonies, imperialists








def Matyas(var):
    x1, x2 = var
    return 0.26*(x1**2 + x2**2) - 0.48*(x1*x2)

alg = Imperialistic()
print(alg.optimize(Matyas))