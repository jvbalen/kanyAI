# binary classifier for sample detection

import matplotlib.pyplot as plt
import itertools
from yaafelib import *
from math import * 

from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA, KernelPCA
import wave
import minirank

import numpy as np
import math
import time
import glob
import sys
import pandas as pd
import ast

# read the feature csv file
featurefile = "features.csv"
# features = np.genfromtxt(featurefile, delimiter=",")
features = np.array([[0,1,0],[1,0,1],[1,1,1],[0,0,1],[0,0,1]])
classfile = "classes.csv"
# classes = np.genfromtxt(classfile, delimiter=",")
classes = np.array([0,1,0,1,1])
test_size = 0.3
random_state = 32
X_train, X_test, y_train, y_test = cross_validation.train_test_split(features, classes, test_size=test_size, random_state=random_state)

# ================================= MACHINE LEARNING ============================================ 
clf = Pipeline([('classification', GradientBoostingClassifier(n_estimators=500, learning_rate=0.1, max_depth=3, random_state=0, verbose=1))])
print "Fitting model..."
clf.fit(X_train, y_train)
print "Testing score..."
scores = clf.score(X_test, y_test)
print "Accuracy: " + str(scores)