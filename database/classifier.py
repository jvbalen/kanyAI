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

import numpy as np

# read the feature csv file
featurefile = "features.csv"
size = 300
nfeats = 17
# features = np.genfromtxt(featurefile, delimiter=",")
features = np.random.rand(size,nfeats)

classfile = "classes.csv"
# classes = np.genfromtxt(classfile, delimiter=",")
classes = np.random.randint(2, size=size)
test_size = 0.3

# ================================= MACHINE LEARNING ============================================ 
clf_rand_state = np.random.randint(100, size=1)[0]
clf = Pipeline([
	('classification', GradientBoostingClassifier(n_estimators=500, learning_rate=0.1, max_depth=3, random_state=clf_rand_state))
	])

# k repeats
k = 10
scores = np.zeros(k)
for i in np.arange(k):
	random_state = np.random.randint(100, size=1)[0]
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(features, classes, test_size=test_size, random_state=random_state)
	print "Fitting model: " + str(i)
	clf.fit(X_train, y_train)
	print "Testing score..."
	score = clf.score(X_test, y_test)
	print "Accuracy: " + str(score)
	scores[i] = score

avg_acc = np.mean(scores)
print "Scores: " + str(scores)
print "Average accuracy: " + str(avg_acc)