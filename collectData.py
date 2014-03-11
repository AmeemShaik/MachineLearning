import numpy as np
import time
import analyzer
from sklearn.cross_validation import KFold

labels = np.loadtxt('tumorData.csv',delimiter=',',dtype='|S6',usecols=(0,))
vectors = np.loadtxt('tumorData.csv',delimiter=',',usecols=(range(1,31)))
kf = KFold(len(labels),n_folds=10,indices = False,shuffle=True,random_state=42)
analyzer.analyzeNB(vectors,labels,'M',kf,"TumorNBResults")
analyzer.analyzeRF(vectors,labels,'M',kf,100,30,"TumorRFResults")
analyzer.analyzeKNN(vectors,labels,'M',kf,100,"TumorKNNResults")

