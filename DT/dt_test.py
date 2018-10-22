# -*- coding: utf-8 -*-
# @Author: vamshi

import pickle
import numpy as np
from dt import DecisionTree
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import dill

headers = ["age","workclass","fnlwgt","education","education-num","marital-status","occupation",
				"relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country","labels"]
dataset_dir = "./data/dataset.pickle"

#Load pickled Test data
with open(dataset_dir,'rb') as f:
	save = pickle.load(f)
	X_train  = save['X_train']
	y_train  = save['y_train']
	X_test   = save['X_test']
	y_test   = save['y_test']
	del save

	print('X_train ', X_train.shape)
	print('y_train', y_train.shape)
	print('X_test ', X_test.shape)

y_train = np.reshape(y_train,(y_train.shape[0],1))
dataset = np.concatenate((X_train,y_train), axis=1)

#Instance of Decision Tree Object
a = DecisionTree(headers,5)
X_tr = dataset[0:6400]
y_tr = y_train[0:6400]

X_val = X_train[6400:8000]
y_val = y_train[6400:8000]
y_val = np.reshape(y_val, (y_val.shape[0],))

t = a.train(dataset)
#Saving Trained Model
dill.dump(a, open("vamshi.model","w"))

v = dill.load(open("vamshi.model"))

y_pred = a.predict(None,X_val)

print y_pred

y_val = np.array(y_val)
y_pred = np.array(y_pred)

acc = np.sum((y_val==y_pred)*1.0)/y_val.shape[0]
print acc

