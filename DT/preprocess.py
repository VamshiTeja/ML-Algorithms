import sys
import os
import numpy as np
import pandas as pd
import pickle
import csv


train_file = "./data/train.csv"
test_file  = "./data/test.csv"
headers = ["age","workclass","fnlwgt","education","education-num","marital-status","occupation",
				"relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country","labels"]


def read_dataset(file_dir,headers,mode):

	df = pd.read_csv(file_dir,names=headers)
	df.head()

	df = pd.DataFrame(df)

	for feature in headers:
		if(type(df[feature][0])!=type(1)):
			discrete_val = np.unique(df[feature])
			for (i,val) in enumerate(discrete_val):
				idx = list(np.where(df[feature]==val))
				#print idx
				for j in idx:
					df[feature][j] = i

	df.to_csv("./data/" + mode + ".csv" )
	df_matrix = df.as_matrix()
	X = df_matrix[:,0:14]

	if mode=="train":
		y = df_matrix[:,14]
	else:
		y = None

	return X,y

X_train, y_train = read_dataset(train_file, headers, "my_train")

X_test, y_test   = read_dataset(test_file, headers, "my_test") 

save_as_pickle = "./data/dataset.pickle"

try:
	f = open(save_as_pickle,'wb')
	save = {
	'X_train'  : X_train,
	'y_train'  : y_train,
	'X_test'   : X_test,
	'y_test'   : y_test
	}
	pickle.dump(save,f,pickle.HIGHEST_PROTOCOL)
	f.close()
except Exception as e:
	print('Unable to save data to ',save_as_pickle,':',e)
	raise

statinfo = os.stat(save_as_pickle)
print('Compressed pickle size',statinfo.st_size)