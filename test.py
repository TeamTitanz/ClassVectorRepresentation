import numpy as np
import pandas as pd
import cPickle
import random
import math
from os import walk

def calculateInputs(vectorList):
    del vectorList[-1]
    del vectorList[0]
    X = np.array(vectorList)
    vectorCount = len(vectorList)
        

    avgInstanceVec = []
    for i in range(300):
        totalComponentValue = 0
        for j in range(vectorCount):
            totalComponentValue += vectorList[j][i]

        avgComponentValue = totalComponentValue/float(vectorCount)
        avgInstanceVec.append(avgComponentValue)

    medianVector = []
    minDistance = 10000000
    for i in range(vectorCount):
        differenceVec = 0
        for j in range(300):
            differenceVec += (vectorList[i][j] - avgInstanceVec[j])**2

        distance = math.sqrt(differenceVec)
        if(minDistance > distance):
            minDistance = distance
            medianVector = vectorList[i]

    medianVectorToReturn = []
    for i in range(300):
        medianVectorToReturn.append(medianVector[i])
        
    return avgInstanceVec, medianVectorToReturn

#in here calculate judge and complaint distances between actual vectors and our model's result vector
classVectors = cPickle.load(open('classes_vectors.p', 'rb'))

finalVector = []
for i in range(11):
    for j in range(300):
        finalVector.append(classVectors[i][j])
        
actualVector  = np.array(finalVector)

df = pd.read_csv('results.csv')
resultVector  = np.array(df['result'])

for i in range(11):
    differenceVec = 0
    for j in range(i*300,(i*300)+300):
        differenceVec += ((actualVector[j]) - resultVector[j])**2

    print(math.sqrt(differenceVec))

######################################################################################################

#Calculate Average Vector of all classes-------------------------------------------------------------
classVectors = cPickle.load(open('classes_vectors.p', 'rb'))
del classVectors[-1]

print(len(classVectors))

finalLabels=[]
for i in range(11):  
    label = np.array(classVectors[i])
    newLabel = []
    for j in range(300):
        temp = label[j]
        newLabel.append(temp)
    finalLabels.append(newLabel)


avgInstanceVec = []
for i in range(300):
    totalComponentValue = 0
    for j in range(11):
        totalComponentValue += finalLabels[j][i]

    avgComponentValue = totalComponentValue/float(11)
    avgInstanceVec.append(avgComponentValue)

#######################################################################################################

#Calculate distance vectors of complaint and judge according to median and average method-------------

##vectorFileNames = filenames = next(walk('E:\MoraHack\FYP\Vectors\Final Law vectors'))[2]
##for i in range(11):
##    
##    vectorList = cPickle.load(open('Final Law vectors\\'+vectorFileNames[i], 'rb'))
##    avgInstanceVec, medianVector = calculateInputs(vectorList)
##    acutalVector = classVectors[i]
##
##    differenceVec = 0
##    for j in range(300):
##        differenceVec += ((acutalVector[j]) - avgInstanceVec[j])**2
##    print(differenceVec)
##
##    differenceVec = 0
##    for j in range(300):
##        differenceVec += ((acutalVector[j]) - medianVector[j])**2
##    print(differenceVec)
##    print('\n')
    
###################################################################################################
#We calculate all class vector average distances--------------------------------------------------

##finalAVGVec = []
##finalMedVec = []
##
##vectorFileNames = filenames = next(walk('E:\MoraHack\FYP\Vectors\Final Law vectors'))[2]
##for i in range(13):
##    if((i != 2) and (i != 7)):
##        vectorList = cPickle.load(open(vectorFileNames[i], 'rb'))
##        avgInstanceVec, medianVector = calculateInputs(vectorList)
##        
##        finalAVGVec.append(avgInstanceVec)
##        finalMedVec.append(medianVector)
##    else:
##        print(vectorFileNames[i])
##
###Calculate Average vector of average vectors of each classes
##avgInstanceAVGVec = []
##for i in range(300):
##    totalComponentValue = 0
##    for j in range(11):
##        totalComponentValue += finalAVGVec[j][i]
##
##    avgComponentValue = totalComponentValue/float(11)
##    avgInstanceAVGVec.append(avgComponentValue)
##
##
###Calculate distance between Class Vector Average vector & Average method vector
##differenceVec = 0
##for j in range(300):
##    differenceVec += ((avgInstanceVec[j]) - avgInstanceAVGVec[j])**2
##
##print(math.sqrt(differenceVec))
##
###Calculate Average vector of Median vectors of each classes
##avgInstanceMedVec = []
##for i in range(300):
##    totalComponentValue = 0
##    for j in range(11):
##        totalComponentValue += finalMedVec[j][i]
##
##    avgComponentValue = totalComponentValue/float(11)
##    avgInstanceMedVec.append(avgComponentValue)
##    
###Calculate distance between Class Vector Average vector & Median method vector
##differenceVec = 0
##for j in range(300):
##    differenceVec += ((avgInstanceVec[j]) - avgInstanceMedVec[j])**2
##
##print(math.sqrt(differenceVec))

#########################################################################################

##df = pd.read_csv('results.csv')
##resultVector  = np.array(df['result'])
##
##ourModelClassVecs = []
##for i in range(11):
##    ourModelClassVec = []
##    for j in range(300):
##        ourModelClassVec.append(resultVector[j+(i*300)])
##    ourModelClassVecs.append(ourModelClassVec)
##
##avgInstanceOurVec = []
##for i in range(300):
##    totalComponentValue = 0
##    for j in range(11):
##        totalComponentValue += ourModelClassVecs[j][i]
##
##    avgComponentValue = totalComponentValue/float(11)
##    avgInstanceOurVec.append(avgComponentValue)
##    
##differenceVec = 0
##for j in range(300):
##    differenceVec += ((avgInstanceVec[j]) - avgInstanceOurVec[j])**2
##
##print(math.sqrt(differenceVec))
