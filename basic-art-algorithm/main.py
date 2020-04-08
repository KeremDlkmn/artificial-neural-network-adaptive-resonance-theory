import numpy as np
import ARTFunctions as art

# Magic Numbers -> %80 #
similarity_coefficient = 0.8

# M1, M2, M3 and P1,P2 and P3 values #
dataset = np.array([
    [1,0,1],
    [1,1,0],
    [0,0,1] ])

inputN,outputM = art.numberOfRowsAndColumns(dataset)

backwardWeight = art.initialForwardDirectionWeightMatrix(inputN,outputM)
forwardWeight  = backwardWeight * art.initialForwardDirectionWeightFormula(inputN)

for i in range(inputN):
    YList = art.calculateYList(inputN,dataset[i],forwardWeight)
    maxYP = art.maxYListInValue(YList)
    PS1 = art.fitnessTestS1(dataset[i])
    newBackwardWeight, newForwardWeight = art.fitnessTestS2(inputN, backwardWeight[i], dataset[i], PS1, similarity_coefficient, maxYP)
    print("New Backward Weight : ", newBackwardWeight)
    print("New Forward Weight  : ", newForwardWeight)