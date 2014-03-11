import numpy as np
import time
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
import csv
from sklearn.metrics import accuracy_score


labels = np.loadtxt('tumorData.csv',delimiter=',',dtype='|S6',usecols=(0,))
vectors = np.loadtxt('tumorData.csv',delimiter=',',usecols=(range(1,31)))
#create a binary mask array
kf = KFold(len(labels),n_folds=10,indices = False,shuffle=True,random_state=42)
gnb = GaussianNB()
nb_total = 0
for train_index, test_index in kf:
    nb = gnb.fit(vectors[train_index],labels[train_index])
    nb_pred = nb.predict(vectors[test_index])
    #print ("NB Accuracy: %3f" % accuracy_score(labels[test_index],nb_pred))
    nb_total+=accuracy_score(labels[test_index],nb_pred)

print ("Average NB accuracy: %3f" % (nb_total/len(kf)))

maxTrees = 100
for numTrees in range(1,maxTrees):
    rf = RandomForestClassifier(n_estimators=numTrees)
    rf_total = 0
    for train_index, test_index in kf:
        rf = rf.fit(vectors[train_index],labels[train_index])
        rf_pred = rf.predict(vectors[test_index])
        rf_total+=accuracy_score(labels[test_index],rf_pred)

    print ("Average RF accuracy: %3f" % (rf_total/len(kf)))
maxFeatures = 31
for features in range(1,maxFeatures):
    rf = RandomForestClassifier(n_estimators=31,max_features=features)
    rf_total = 0
    for train_index, test_index in kf:
        rf = rf.fit(vectors[train_index],labels[train_index])
        rf_pred = rf.predict(vectors[test_index])
        rf_total+=accuracy_score(labels[test_index],rf_pred)
    print
    print ("Average RF accuracy: %3f" % (rf_total/len(kf)))
maxNeighbors = 100
for numNeighbors in range(1,maxNeighbors):
    neigh = KNeighborsClassifier(n_neighbors=numNeighbors).fit(vectors[train_index],labels[train_index])
    neigh_total = 0
    for train_index, test_index in kf:
        neigh_pred = neigh.fit(vectors[train_index],labels[train_index])
        neigh_pred = neigh.predict(vectors[test_index])
        neigh_total+=accuracy_score(labels[test_index],neigh_pred)
    print ("Average KNN accuracy: %3f" % (neigh_total/len(kf)))
