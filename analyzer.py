import numpy as np
import time
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
import csv
from sklearn.metrics import accuracy_score


def analyzeNB(vectors,labels,posLabel,kf,filename):
    bayesWriter = csv.writer(open(filename+".csv","wb"))
    gnb = GaussianNB()
    bayesWriter.writerow(["Iteration","Accuracy","Precision","Sensitivity","Specificity"])
    total = {'accuracy':0,'precision':0,"sensitivity":0,"specificity":0}
    i = 0
    for train_index, test_index in kf:
        nb = gnb.fit(vectors[train_index],labels[train_index])
        nb_pred = nb.predict(vectors[test_index])
        metrics = __getMetrics(labels[test_index],nb_pred,posLabel)
        accuracy = metrics['accuracy'] 
        precision = metrics['precision']
        sensitivity = metrics['sensitivity']
        specificity = metrics['specificity']
        total['accuracy']+=accuracy
        total['precision']+=precision
        total['sensitivity']+=sensitivity
        total['specificity']+=specificity
        bayesWriter.writerow([i,accuracy,precision,sensitivity,specificity])
        i+=1 
    bayesWriter.writerow(["Averages",total['accuracy']/len(kf),total['precision']/len(kf),total['sensitivity']/len(kf),total['specificity']/len(kf)])



def analyzeRF(vectors,labels,posLabel,kf,maxTrees,maxFeatures,filename):
    rfWriter = csv.writer(open(filename+".csv","wb"))
    rfWriter.writerow(["Number of trees", "Average accuracy","Average Precision","Average Sensitivity","Average Specificity"])
    for numTrees in range(1,maxTrees+1):
        randomForest = RandomForestClassifier(n_estimators=numTrees)
        total = {'accuracy':0,'precision':0,"sensitivity":0,"specificity":0}
        for train_index, test_index in kf:
            rf = randomForest.fit(vectors[train_index],labels[train_index])
            rf_pred = rf.predict(vectors[test_index])
            metrics = __getMetrics(labels[test_index],rf_pred,posLabel)
            accuracy = metrics['accuracy'] 
            precision = metrics['precision']
            sensitivity = metrics['sensitivity']
            specificity = metrics['specificity']
            total['accuracy']+=accuracy
            total['precision']+=precision
            total['sensitivity']+=sensitivity
            total['specificity']+=specificity
        rfWriter.writerow([numTrees,total['accuracy']/len(kf),total['precision']/len(kf),total['sensitivity']/len(kf),total['specificity']/len(kf)])

    for features in range(1,maxFeatures+1):
        randomForest = RandomForestClassifier(n_estimators=30,max_features=features)
        rfWriter.writerow(["Number of features","Average accuracy","Average Precision","Average Sensitivity", "Average Specificity"])
        total = {'accuracy':0,'precision':0,"sensitivity":0,"specificity":0}
        for train_index, test_index in kf:
            rf = randomForest.fit(vectors[train_index],labels[train_index])
            rf_pred = rf.predict(vectors[test_index])
            metrics = __getMetrics(labels[test_index],rf_pred,posLabel)
            accuracy = metrics['accuracy'] 
            precision = metrics['precision']
            sensitivity = metrics['sensitivity']
            specificity = metrics['specificity']
            total['accuracy']+=accuracy
            total['precision']+=precision
            total['sensitivity']+=sensitivity
            total['specificity']+=specificity
        rfWriter.writerow([features,total['accuracy']/len(kf),total['precision']/len(kf),total['sensitivity']/len(kf),total['specificity']/len(kf)])

         
def analyzeKNN(vectors,labels,posLabel,kf,maxNeighbors,filename):
    knnWriter = csv.writer(open(filename+".csv","wb"))
    knnWriter.writerow(["Number of neighbors","Average accuracy","Average Precision","Average Sensitivity", "Average Specificity"])
    for numNeighbors in range(1,maxNeighbors+1):
        knn = KNeighborsClassifier(n_neighbors=numNeighbors)
        total = {'accuracy':0,'precision':0,"sensitivity":0,"specificity":0}
        for train_index, test_index in kf:
            neigh = knn.fit(vectors[train_index],labels[train_index])
            neigh_pred = neigh.predict(vectors[test_index])
            metrics = __getMetrics(labels[test_index],neigh_pred,posLabel)
            accuracy = metrics['accuracy'] 
            precision = metrics['precision']
            sensitivity = metrics['sensitivity']
            specificity = metrics['specificity']
            total['accuracy']+=accuracy
            total['precision']+=precision
            total['sensitivity']+=sensitivity
            total['specificity']+=specificity
        knnWriter.writerow([numNeighbors,total['accuracy']/len(kf),total['precision']/len(kf),total['sensitivity']/len(kf),total['specificity']/len(kf)])
        

def __getMetrics(trueLabels,predLabels,posLabel):
    truePos=trueNeg=falsePos=falseNeg = 0
    for i in range(len(trueLabels)):
       if(trueLabels[i]==posLabel):
           if(predLabels[i]==posLabel):
               truePos+=1
           else:
               falseNeg+=1
       else:
           if(predLabels[i]==posLabel):
               falsePos+=1
           else:
               trueNeg+=1
    results={}
    results['accuracy']=(truePos+trueNeg)/float(truePos+falsePos+falseNeg+trueNeg)
    results['precision']=(truePos)/float(truePos+falsePos)
    results['sensitivity'] =(truePos)/float(truePos+falseNeg)
    results['specificity'] =(trueNeg)/float(trueNeg+falsePos)
    return results 
