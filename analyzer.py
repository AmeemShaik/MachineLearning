import numpy as np
import time
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
import csv
from sklearn.metrics import accuracy_score


def analyzeNB(vectors,labels,kf,filename):
    bayesWriter = csv.writer(open(filename+".csv","wb"))
    gnb = GaussianNB()
    total = 0
    i = 0
    for train_index, test_index in kf:
        nb = gnb.fit(vectors[train_index],labels[train_index])
        nb_pred = nb.predict(vectors[test_index])
        accuracy = accuracy_score(labels[test_index],nb_pred)
        total+=accuracy
        bayesWriter.writerow([i,accuracy])
        i+=1 
    bayesWriter.writerow(["Average Accuracy",total/len(kf)])



def analyzeRF(vectors,labels,kf,maxTrees,maxFeatures,filename):
    rfWriter = csv.writer(open(filename+".csv","wb"))
    rfWriter.writerow(["Number of trees", "Average accuracy"])
    for numTrees in range(1,maxTrees+1):
        randomForest = RandomForestClassifier(n_estimators=numTrees)
        rf_total = 0
        for train_index, test_index in kf:
            rf = randomForest.fit(vectors[train_index],labels[train_index])
            rf_pred = rf.predict(vectors[test_index])
            rf_total+=accuracy_score(labels[test_index],rf_pred)

        rfWriter.writerow([numTrees,rf_total/len(kf)]) 

    rfWriter.writerow(["Number of features","Average accuracy"])
    for features in range(1,maxFeatures+1):
        randomForest = RandomForestClassifier(n_estimators=30,max_features=features)
        total = 0
        for train_index, test_index in kf:
            rf = randomForest.fit(vectors[train_index],labels[train_index])
            rf_pred = rf.predict(vectors[test_index])
            total+=accuracy_score(labels[test_index],rf_pred)
        rfWriter.writerow([features,total/len(kf)])
         
         
def analyzeKNN(vectors,labels,kf,maxNeighbors,filename):
    knnWriter = csv.writer(open(filename+".csv","wb"))
    knnWriter.writerow(["Number of neighbors","Average accuracy"])
    for numNeighbors in range(1,maxNeighbors+1):
        knn = KNeighborsClassifier(n_neighbors=numNeighbors)
        total = 0
        for train_index, test_index in kf:
            neigh = knn.fit(vectors[train_index],labels[train_index])
            neigh_pred = neigh.predict(vectors[test_index])
            total+=accuracy_score(labels[test_index],neigh_pred)
        knnWriter.writerow([numNeighbors, total/len(kf)])
