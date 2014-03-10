import numpy as np
import time
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validaiton import KFold
import csv

labels = np.loadtxt('data.csv',delimiter=',',dtype='|S6',usecols=(0,))
vectors = np.loadtxt('data.csv',delimiter=',',usecols=(range(1,31)))
vectors_train, vectors_test, labels_train, labels_test = train_test_split(vectors, labels, test_size=0.20, random_state=42)
start = time.time()
gnb = GaussianNB()
y_pred = gnb.fit(vectors_train, labels_train).predict(vectors_test)
end = time.time()
print ("NB Time: %.5f" % (end - start))
print("NB Number of mislabeled points : %d" % (labels_test != y_pred).sum())

start = time.time()
rf = RandomForestClassifier(n_estimators=5)
rf_pred = rf.fit(vectors_train, labels_train).predict(vectors_test)
end = time.time()
print ("RF Time: %.5f" % (end - start))
print("RF Number of mislabeled points : %d" % (labels_test != rf_pred).sum())

start = time.time()
neigh = KNeighborsClassifier(n_neighbors=100).fit(vectors_train, labels_train)
neigh_pred = neigh.predict(vectors_test)
end = time.time()
print ("KNN Time: %.5f" % (end - start))
print("KNN Number of mislabeled points : %d" % (labels_test != neigh_pred).sum())

#print y_pred
#print rf_pred
#print neigh_pred
#print labels_test