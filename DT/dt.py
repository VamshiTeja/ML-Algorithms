# -*- coding: utf-8 -*-
# @Author: vamshi

import numpy as np

class DecisionTree():

	'''
		Decision Tree classifier class based on univariate splits
	'''

	def __init__(self,headers,min_size):

		import numpy
		import csv
		import pandas
		self.pd = pandas
		self.np = numpy
		self.csv = csv
		self.headers = headers
		self.min_size = min_size

	def read_dataset(self,file_dir,mode):
		'''
			Function to read the dataset taking its file directory using pandas module
		'''
		df = self.pd.read_csv(file_dir,names=self.headers)
		df.head()

		df = self.pd.DataFrame(df)

		#Here in the dataset only string type attributes had '?' as one of the attribute, 
		#I considered each '?' as new attribute for every column 
		#
		for feature in self.headers:
			if(type(df[feature][0])!=type(1)):
				discrete_val = self.np.unique(df[feature])
				for (i,val) in enumerate(discrete_val):
					idx = list(self.np.where(df[feature]==val))
					#print idx
					for j in idx:
						df[feature][j] = i

		df_matrix = df.as_matrix()
		X = df_matrix[:,0:14]

		if mode=="train":
			y = df_matrix[:,14]
		else:
			y = None

		return X,y

	def read_dataset_csv(self,file_dir,mode):
		'''
			Function to read the dataset taking its file directory using csv module
		'''
		rows = []
		with open(file_dir,'r') as csvfile:

			csvreader = self.csv.reader(csvfile)
			for row in csvreader:
				rows.append(row)

			data = np.array(rows)
			dataset = np.zeros(shape = data.shape,dtype=np.int)

			#remove whitespaces at start and end of string
			print data.shape
			for i in range(data.shape[0]):
				for j in range(data.shape[1]):
					data[i][j] = data[i][j].strip()   

			idx =[]
			for i in range(data.shape[1]):
				if(data[0][i].isdigit()):
					for j in range(data.shape[0]):
						dataset[j][i] = (data[j][i]).astype(np.int64)
				else:
					idx.append(i)
			
			for f in idx:
				discrete_val = np.unique(data[:,f])
				for (i,val) in enumerate(discrete_val):
					idx = list(np.where(data[:,f]==val))
					for j in idx:
						dataset[j,f] = i

		X = dataset[:,0:14]

		if mode=="train":
			y = dataset[:,14]
		else:
			y =None

		return X,y


	def group_split(self,index,value,dataset):
		'''
			Basic Function to split(univariate) according  and value of a attribute 
		'''
		l, r = [], []
		for row in dataset:
			if(row[index]<value):
				l.append(row)
			else:
				r.append(row)
		return l,r


	def gini_idx(self,groups,classes):
		'''
			Function to calculate gini impurity at a node
		'''
		n_reached = float(sum([len(group) for group in groups]))
		g = 0.0
		for group in groups:
			size = float(len(group))
			if(size==0):
				continue
			s = 0.0
			for c in classes:
				prob = [row[-1] for row in group].count(c)/size
				s = s + prob*prob
			g = g + (1-s)*size/n_reached
		return g

	def do_split(self,dataset): 
		'''
			Function to choose a split minimizing gini impurity at a node 
		'''
		dataset = np.array(dataset)
		classes = self.np.unique(dataset[:,-1])
		split_idx, split_val, split_score, split_groups = 12,5000, 10, None
		tmp = -1
		for idx in range((dataset.shape[1]-1)):
			for row in dataset:
				if(row[idx]==tmp):
					continue
				else:
					groups = self.group_split(idx,row[idx],dataset)
					gini = self.gini_idx(groups, classes)
					tmp = row[idx]
					#print('A%d > %.2f Gini = %.2f'%(idx,row[idx],gini))
					if(gini<split_score):
						split_idx,split_val,split_score,split_groups = idx,row[idx],gini,groups
		return {'index':split_idx,'value':split_val,'groups':split_groups}

	def end_node(self,group):
		'''
			Function to declare a node as end node.
			At this node it outputs c
		'''
		classes_at_group =[r[-1] for r in group]
		unique, counts = self.np.unique(classes_at_group,return_counts = True)
		idx = self.np.argmax(counts)
		return idx

	def recsplit(self,node,min_size):
		'''
			Function to recursively split the node. 
			It stops splitting when a group has less than req members.
		'''
		l,r = node['groups']
		del(node['groups'])

		# Checks if length of groups has zero or nor 
		if (len(l)==0):
			node['r'] = node['l'] = self.end_node(r)
		elif(len(r)==0):
			node['r'] = node['l'] = self.end_node(l)
		else:
			# Checks whether group has min no. of elements or not
			if(len(l)<min_size):
				node['l'] = self.end_node(l)
			else:
				#if not it splits recursively
				node['l'] = self.do_split(l)
				self.recsplit(node['l'],min_size)

			if(len(r)<min_size):
				node['r'] = self.end_node(r)
			else:
				node['r'] = self.do_split(r)
				self.recsplit(node['r'], min_size)

	def train(self,trainset):
		'''
			Function to train the decision tree
		'''
		root = self.do_split(trainset)
		self.recsplit(root, self.min_size)
		self.root = root
		return root

	def predict_row(self,node,row):
		'''
			Function to predict output for a row 
		'''
		if(row[node['index']]<node['value']):
			if(isinstance(node['l'],dict)):
				return self.predict_row(node['l'], row)
			else:
				return node['l']
		else:
			if(isinstance(node['r'],dict)):		
				return self.predict_row(node['r'], row)
			else:
				return node['r']

	def predict(self,test_file=None,x=None):
		'''
			test_file : Takes location of test_file, this function expects test_file which is similar to that of train file
			x         : If test_file is not specified(None), it accepts x(x_trai ) 
		'''
		if test_file is not None:
			data_test,_ = self.read_dataset(test_file, "test")
		if x is not None:
			data_test = x

		predictions = []
		for row in data_test:
			y = self.predict_row(self.root, row)
			predictions.append(y)
		self.predictions = predictions
		return predictions