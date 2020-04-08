import numpy as np

def initialForwardDirectionWeightFormula(N):
    return (1) / (1 + N)

def initialForwardDirectionWeightMatrix(N,M):
    return np.ones((N, M), dtype=np.uint8())

def previousBackwardWeight(forwardWeightMatrix):
    return forwardWeightMatrix

def numberOfRowsAndColumns(matrix):
    return matrix.shape[0], matrix.shape[1]

def calculateYList(N,inputList,forwardWeightList):
    outputListY = [0,0,0]
    for i in range(N):
        for j in range(N):
            outputListY[i] = outputListY[i] + (forwardWeightList[j] * inputList[j])
    return outputListY[0]

def maxYListInValue(YList):
    return max(YList)

def fitnessTestS1(inputList):
    s1 = 0
    for i in inputList:
        s1 = s1 + i
    return s1

def fitnessTestS2(N,weightList,inputList,S1,p,YMax):
    S2 = [0,0,0]
    for i in range(N):
        for j in range(N):
            S2[i] = S2[i] + (weightList[j] * inputList[j])
    s2 = S2[0] / S1
    if(int(s2) >= p):
        return updateForwardAndBackwardWeights(weightList,inputList,YMax)
    else:
        return 0

def updateForwardAndBackwardWeights(weightList,inputList,YMax):
    newBackwardWeight  = [0,0,0]
    newForwardWeight   = [0,0,0]

    for index in range(len(newBackwardWeight)):
        newBackwardWeight[index] = newBackwardWeight[index] + (weightList[index] * inputList[index])

    seriesOperationList = [0,0,0]

    for i in range(len(newForwardWeight)):
        for j in range(len(newForwardWeight)):
            seriesOperationList[i] = seriesOperationList[i] + (newBackwardWeight[i] * inputList[j])
        newForwardWeight[i] = newForwardWeight[i] + ( (newBackwardWeight[i] * inputList[i]) / (YMax + (seriesOperationList[i]) ) )
    return newBackwardWeight,newForwardWeight