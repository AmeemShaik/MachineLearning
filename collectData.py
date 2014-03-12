import numpy as np
import pandas as pd
import time
import analyzer
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import KFold
from sklearn import preprocessing

'''labels = np.loadtxt('tumorData.csv',delimiter=',',dtype='|S6',usecols=(0,))
vectors = np.loadtxt('tumorData.csv',delimiter=',',usecols=(range(1,31)))
kf = KFold(len(labels),n_folds=10,indices = False,shuffle=True,random_state=42)
analyzer.analyzeNB(vectors,labels,'M',kf,"TumorNBResults")
analyzer.analyzeRF(vectors,labels,'M',kf,100,30,"TumorRFResults")
analyzer.analyzeKNN(vectors,labels,'M',kf,100,"TumorKNNResults")
'''
data = pd.read_csv('CensusData.csv')
columns = ['Age','workclass','fnlwgt','education','assoc-voc','marital','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country']
labels = np.array(data['Label'])
rawVectors = data[list(columns)].values
le = preprocessing.LabelEncoder()
vectors=[]
for c in rawVectors.T:
    categories = set(c)
    le.fit(list(categories))
    vectors.append(le.transform(c))
vectors = np.array(vectors).T
kf = KFold(len(labels),n_folds=10,indices = False,shuffle=True,random_state=42)
#analyzer.analyzeNB(vectors,labels,' >50K',kf,'SalaryNBResults')
#analyzer.analyzeRF(vectors,labels,' >50K',kf,100,13,'SalaryRFResults')
analyzer.analyzeKNN(vectors,labels,' >50K',kf,100,'SalaryKNNResults')
