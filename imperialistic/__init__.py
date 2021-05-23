import copy
import random

import numpy as np


class Imperialistic:
    def __init__(self, **kwargs):
        self.N = kwargs.get('N', 10)
        self.b_low = kwargs.get('b_low', -10)
        self.b_up = kwargs.get('b_up', 10)
        self.if_min = kwargs.get('if_min', True)
        self.beta = kwargs.get('beta', 2)
        self.gamma = kwargs.get('gamma', 0.25)
        self.xi = kwargs.get('xi', 0.1)
        self.numImp = kwargs.get('numberImp', int(0.3*self.N))
        self.numCol = int(self.N - self.numImp)
        self.return_all_best = kwargs.get('return_all_best', False)


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
        countries = X[self.numImp:]
        # Normalized cost
        normCost = costCountry[:self.numImp] - np.max(costCountry[:self.numImp])
        # Power of every imperialist
        sumImp = np.sum(normCost[:self.numImp])
        powerImp = np.abs(normCost/sumImp)
        # Initial number of colonies of an empire
        numColOfImp = np.round(self.numCol*powerImp)
        # Divide colonies in random way
        colonies = self.__divideColonies(countries, numColOfImp)
        imperialists, colonies = self.__removeEmptyEmpireFirst(imperialists, colonies)
        while len(imperialists) > 1:

            # Move the colonies toward their revelant imperialst
            colonies = self.__moveColonies(colonies, imperialists, d)

            # Check if the imperialist have better positions than a colony
            colonies, imperialists = self.__checkPosition(colonies, imperialists, f)

            # Total cost of an empire (total power)
            totalCostEmp = self.__totalPowerOfEmpire(colonies, imperialists, f)

            # Normalized total cost
            normTotalCostEmp = totalCostEmp - np.max(totalCostEmp)

            # Possession probability of each empire
            P = np.abs(normTotalCostEmp / np.sum(normTotalCostEmp))
            R = np.random.uniform(0, 1, size=np.size(P))
            D = P - R
            maxImp = np.argmin(D)
            minImp = np.argmax(D)

            # Imperialistic Competition
            colonies, imperialists = self.__competition(colonies, imperialists, maxImp, minImp, f)
            imperialists, colonies = self.__removeEmptyEmpireSecond(imperialists, colonies, D)
        return imperialists[0]


    def __divideColonies(self, colonies, numColOfImp):
        divColonies = []
        sumCol = 0
        np.random.shuffle(colonies)
        if sum(numColOfImp) != self.numCol:
            if sum(numColOfImp) > self.numCol:
                numColOfImp[0] -= sum(numColOfImp) - self.numCol
            else:
                numColOfImp[0] += self.numCol - sum(numColOfImp)
        for num in numColOfImp:
            divColonies.append(colonies[sumCol:sumCol+int(num)])
            sumCol += int(num)
        return np.array(divColonies)

    def __moveColonies(self, colonies, imperialists, d):
        for imp in range(len(imperialists)):
            for col in range(len(colonies[imp])):
                for i in range(d):
                    if (colonies[imp][col][i] < self.b_up) & (colonies[imp][col][i] > self.b_low):
                        diff = imperialists[imp][i] - colonies[imp][col][i]
                        #bdif = self.beta*diff
                        if diff > 0:
                            colonies[imp][col][i] += np.random.uniform(0, self.beta*diff, 1) + np.random.uniform(-self.gamma, self.gamma, 1)
                        elif diff < 0:
                            colonies[imp][col][i] += np.random.uniform(self.beta*diff, 0, 1) + np.random.uniform(-self.gamma, self.gamma, 1)
                        elif diff == 0:
                            colonies[imp][col][i] += np.random.uniform(-self.gamma, self.gamma, 1)
                    if colonies[imp][col][i] > self.b_up:
                        colonies[imp][col][i] = self.b_up - np.random.uniform(0, self.beta, 1)
                    elif colonies[imp][col][i] < self.b_low:
                        colonies[imp][col][i] = self.b_low + np.random.uniform(0, self.beta, 1)
        return colonies

    def __checkPosition(self, colonies, imperialists, f):
        for i in range(len(imperialists)):
            imperialist = copy.copy(imperialists[i])
            for j in range(len(colonies[i])):
                if self.if_min == True:
                    if f(imperialists[i]) > f(colonies[i][j]):
                        imperialists[i] = colonies[i][j]
                        colonies[i][j] = imperialist
                if self.if_min == False:
                    if f(imperialists[i]) < f(colonies[i][j]):
                        imperialists[i] = colonies[i][j]
                        colonies[i][j] = imperialist
        return colonies, imperialists

    def __cost(self, colonies, imperialists, f):
        costOfImperialist = np.zeros(len(imperialists))
        costOfEmpire = np.zeros(len(imperialists))
        for i in range(len(imperialists)):
            costOfImperialist[i] = f(imperialists[i])
            for j in range(len(colonies[i])):
                costOfEmpire[i] += f(colonies[i][j])
        return costOfEmpire, costOfImperialist

    def __totalPowerOfEmpire(self, colonies, imperialists, f):
        totalPowerEmp = []
        costOfEmpire, costOfImperialist = self.__cost(colonies, imperialists, f)
        for i in range(len(colonies)):
            if colonies[i].size == 0:
                meanColony = 0
            else:
                meanColony = np.mean(costOfEmpire[i])
            totalPowerEmp.append(costOfImperialist[i]+self.xi*meanColony)
        return totalPowerEmp

    def __competition(self, colonies, imperialists, bestImp, weakestImp, f):
        if len(colonies[weakestImp]) == 0:
            weakest = imperialists[weakestImp]
            colonies[bestImp] = np.reshape(np.append(colonies[bestImp], weakest), (-1, 2))
            imperialists = np.delete(imperialists, weakestImp, axis=0)
            colonies = np.delete(colonies, weakestImp, axis=0)
        else:
            weakest = colonies[weakestImp][0]
            for i in range(1, len(colonies[weakestImp])):
                if self.if_min == True:
                    if f(weakest) < f(colonies[weakestImp][i]):
                        weakest = colonies[weakestImp][i]
            colonies[bestImp] = np.reshape(np.append(colonies[bestImp], weakest), (-1,2))
            colonies[weakestImp] = np.array([colony for colony in colonies[weakestImp] if not np.all(colony == weakest)])
        return colonies, imperialists

    def __removeEmptyEmpireFirst(self, imperialists, colonies):
        toImp = 0
        lenCol = len(colonies)
        i = 0
        while i < lenCol:
            if len(colonies[i]) == 0:
                colonies[toImp] = np.append(colonies[toImp], [imperialists[i]], axis=0)
                imperialists = np.delete(imperialists, i, axis=0)
                colonies = np.delete(colonies, i, axis=0)
                toImp = toImp+1 % len(imperialists)
                i-=1
                lenCol-=1
            i+=1
        return imperialists, colonies


    def __removeEmptyEmpireSecond(self, imperialists, colonies, P):
        argP = np.argsort(P)
        toImp = 0
        lenCol = len(colonies)
        i = 0
        while i < lenCol:
            if len(colonies[i]) == 0:
                colonies[argP[toImp]] = np.append(colonies[argP[toImp]], [imperialists[i]], axis=0)
                imperialists = np.delete(imperialists, i, axis=0)
                colonies = np.delete(colonies, i, axis=0)
                toImp = toImp+1 % len(imperialists)
                argP = np.delete(argP, i, axis=0)
                i-=1
                lenCol-=1
            i+=1
        return imperialists, colonies