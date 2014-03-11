import numpy as np
import time
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
import csv
from sklearn.metrics import accuracy_score


bayesWriter = csv.writer(open("NBResults.csv","wb"))
rfWriter = csv.writer(open("RFResults.csv","wb"))
knnWriter = csv.writer(open("KNNResults.csv","wb"))


labels = np.loadtxt('tumorData.csv',delimiter=',',dtype='|S6',usecols=(0,))
vectors = np.loadtxt('tumorData.csv',delimiter=',',usecols=(range(1,31)))
#create a binary mask array
kf = KFold(len(labels),n_folds=10,indices = False,shuffle=True,random_state=42)
gnb = GaussianNB()
nb_total = 0
i = 0
for train_index, test_index in kf:
    nb = gnb.fit(vectors[train_index],labels[train_index])
    nb_pred = nb.predict(vectors[test_index])
    accuracy = accuracy_score(labels[test_index],nb_pred)
    nb_total+=accuracy
    bayesWriter.writerow([i,accuracy])
    i+=1 
bayesWriter.writerow(["Average Accuracy",nb_total/len(kf)])

rfWriter.writerow(["Number of trees", "Average accuracy"])
maxTrees = 100
for numTrees in range(1,maxTrees+1):
    rf = RandomForestClassifier(n_estimators=numTrees)
    rf_total = 0
    for train_index, test_index in kf:
        rf = rf.fit(vectors[train_index],labels[train_index])
        rf_pred = rf.predict(vectors[test_index])
        rf_total+=accuracy_score(labels[test_index],rf_pred)

    rfWriter.writerow([numTrees,rf_total/len(kf)]) 

rfWriter.writerow(["Number of features","Average accuracy"])
maxFeatures = 31
for features in range(1,maxFeatures):
    rf = RandomForestClassifier(n_estimators=31,max_features=features)
    rf_total = 0
    for train_index, test_index in kf:
        rf = rf.fit(vectors[train_index],labels[train_index])
        rf_pred = rf.predict(vectors[test_index])
        rf_total+=accuracy_score(labels[test_index],rf_pred)
    rfWriter.writerow([features,rf_total/len(kf)])
    
knnWriter.writerow(["Number of neighbors","Average accuracy"])
maxNeighbors = 100
for numNeighbors in range(1,maxNeighbors+1):
    neigh = KNeighborsClassifier(n_neighbors=numNeighbors).fit(vectors[train_index],labels[train_index])
    neigh_total = 0
    for train_index, test_index in kf:
        neigh_pred = neigh.fit(vectors[train_index],labels[train_index])
        neigh_pred = neigh.predict(vectors[test_index])
        neigh_total+=accuracy_score(labels[test_index],neigh_pred)
    knnWriter.writerow([numNeighbors, neigh_total/len(kf)])
