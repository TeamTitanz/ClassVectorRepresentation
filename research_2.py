import numpy as np
import pandas as pd
import cPickle
from os import walk
import random
import math
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.linear_model import Perceptron


def calculateInputs(vectorList):
    del vectorList[-1]
    del vectorList[0]
    X = np.array(vectorList)
    vectorCount = len(vectorList)

    kmeans = KMeans(n_clusters=2)
    kmeans.fit(X)

    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_

    cluster_1 = []
    cluster_2 = []
    for i in range(vectorCount):
        if(labels[i] == 0):
            cluster_1.append(vectorList[i])
        else:
            cluster_2.append(vectorList[i])


    clf = svm.SVC(gamma = 0.001, C=100, kernel='linear')
    clf.fit(X,labels)


    avgCluster_1Vec = []
    for i in range(300):
        totalComponentValue = 0
        for j in range(len(cluster_1)):
            totalComponentValue += cluster_1[j][i]

        avgComponentValue = totalComponentValue/float(len(cluster_1))
        avgCluster_1Vec.append(avgComponentValue)

    avgCluster_2Vec = []
    for i in range(300):
        totalComponentValue = 0
        for j in range(len(cluster_2)):
            totalComponentValue += cluster_2[j][i]

        avgComponentValue = totalComponentValue/float(len(cluster_2))
        avgCluster_2Vec.append(avgComponentValue)
        

    avgInstanceVec = []
    for i in range(300):
        totalComponentValue = 0
        for j in range(vectorCount):
            totalComponentValue += vectorList[j][i]

        avgComponentValue = totalComponentValue/float(vectorCount)
        avgInstanceVec.append(avgComponentValue)

    avgSVs = []
    for i in range(300):
        totalComponentValue = 0
        for j in range(len(clf.support_vectors_)):
            totalComponentValue += clf.support_vectors_[j][i]

        avgComponentValue = totalComponentValue/float(len(clf.support_vectors_))
        avgSVs.append(avgComponentValue)

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
        
    return avgSVs, avgInstanceVec, avgCluster_1Vec, avgCluster_2Vec, medianVectorToReturn


vectorFileNames = filenames = next(walk('E:\MoraHack\FYP\Vectors\Final Law vectors'))[2]

finalAvgSVs = []
finalAvgInstanceVec = []
finalAvgCluster_1Vec = []
finalAvgCluster_2Vec = []
finalMedianVector = []

for i in range(12):
    if(i != 6):
        vectorList = cPickle.load(open(vectorFileNames[i], 'rb'))
        avgSVs, avgInstanceVec, avgCluster_1Vec, avgCluster_2Vec, medianVector = calculateInputs(vectorList)
        
        finalAvgSVs += avgSVs
        finalAvgInstanceVec += avgInstanceVec
        finalAvgCluster_1Vec += avgCluster_1Vec
        finalAvgCluster_2Vec += avgCluster_2Vec
        finalMedianVector += medianVector



# Data
d = np.array([finalAvgSVs, finalAvgInstanceVec, finalAvgCluster_1Vec, finalAvgCluster_2Vec, finalMedianVector])

# Labels
classVectors = cPickle.load(open('../classes_vectors.p', 'rb'))
del classVectors[-1]

finalLabels=[]
for i in range(11):  
    label = np.array(classVectors[i])
    newLabel = []
    for j in range(300):
        temp = label[j]
        newLabel.append(temp)
    finalLabels += newLabel

# rotate the data 180 degrees
d90 = np.rot90(d)
d90 = np.rot90(d90)
d90 = np.rot90(d90)

df1 = pd.DataFrame(d90)

df2 = pd.DataFrame(finalLabels)

df3 = df1.join(df2, lsuffix='_df1', rsuffix='_df2')

df1.to_csv("perceptron_input.csv")
df2.to_csv("perceptron_output.csv")
df3.to_csv("perceptron_combined.csv")

