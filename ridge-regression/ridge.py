# -*- coding: utf-8 -*-
# @Author: vamshi
# @Date:   2017-11-13 16:11:10
# @Last Modified by:   vamshi
# @Last Modified time: 2017-11-17 23:12:38

import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

def create_req(m,d,l=10):
	'''
		This function creates dataset with m instances and d dimensions
	'''
	np.random.seed(422)
	X = np.random.rand(m, d)
	theta = np.zeros(shape=(d,1))
	r = np.random.randint(0, 2, l)
	for (idx,val) in enumerate(r):
		if(val==0):
			theta[idx] = -l
		else:
			theta[idx] = l
	eps = np.random.randn(m, 1)
	y = np.dot(X,theta) + eps
	return X, y

def ridge_reg(lamb,X,y):
	'''
		Input : Takes regularisation co-eff and training data
		Output: Returns weights of obtained regressor
	'''
	I = np.identity(X.shape[1])
	A = np.matmul(X.T, X) + lamb*I
	A = np.linalg.inv(A)
	b = np.matmul(X.T, y)
	theta_est = np.matmul(A, b)
	#print theta_est
	return theta_est

def error_ridge(arg):
	'''
		Computes squares error for ridge regression
		Input: Takes list of arguments accepts regularisation coefficient 
		This function is given to optimzer to get lambda that minimizes error
	'''
	x  = arg[0]
	theta_est = ridge_reg(x, X_tr, y_tr)
	y_est = np.matmul(X_val, theta_est)
	error = np.square(np.sum(y_val-y_est)/y_val.shape[0])
	return error

def err_lasso(arg):
	'''
		Computes squares error for Lasso
		Input: Takes list of arguments accepts regularisation coefficient 
		This function is given to optimzer to get lambda that minimizes error
	'''
	lamb = arg[0]
	clf = Lasso(alpha=lamb)
	clf.fit(X_tr, y_tr)
	y_est = clf.predict(X_val)
	err = np.square(np.sum(y_val-y_est)/y_val.shape[0])
	return err	

def record_ridge(lamb,thresholds):
	'''
		This function records no of weights that non-zero based on threshold for ridge regression
	'''
	r = []
	theta_est = ridge_reg(lamb, X_tr, y_tr)
	for thresh in thresholds:
		count = 0
		for val in theta_est[10:]:
			if(val>=thresh):
				count = count +1
		r.append(count)	
	return r

def record_lasso(lamb,thresholds):
	'''
		This function records no of weights that non-zero based on threshold for lasso
	'''
	r = []
	clf =  Lasso(alpha=lamb)
	clf.fit(X_tr, y_tr)
	theta_est_lasso = clf.coef_
	#print theta_est_lasso
	for thresh in thresholds:
		count = 0
		for val in theta_est_lasso[10:]:
			if(val>=thresh):
				count = count +1
		r.append(count)	
	return r

#Create Data
X, y = create_req(150, 75)

#Split the data into train, validation, test sets
X_tr, X_test, y_tr, y_test = train_test_split(X,y,test_size = 0.333,random_state=42)
X_tr, X_val, y_tr, y_val   = train_test_split(X_tr,y_tr,test_size=0.2, random_state=42)

#initial guess of lambda for optimizer
init_guess = 0.1

#Minimizes error as a function of lambda
result = minimize(error_ridge, init_guess)
opt_lamb_r =  result['x']

result = minimize(err_lasso, init_guess)
opt_lamb_l = result['x']

#calculating rmse error for test test for ridge
w = ridge_reg(opt_lamb_r, X_tr, y_tr)
y_pred = np.matmul(X_test, w)
error = np.sqrt(np.sum(np.matmul(y_pred-y_test,(y_pred-y_test).T)))
print error

#calculating rmse error for test test for lasso
r = Lasso(alpha=opt_lamb_r)
r.fit(X_tr, y_tr)
y_pred = r.predict(X_test)
error = np.sqrt(np.sum(np.matmul(y_pred-y_test,(y_pred-y_test).T)))
print error


#thresholds to check for
threshs = [0.000001*i for i in range(1,1000,5)]
r_ridge = record_ridge(opt_lamb_r, threshs)
print r_ridge

#Same repeated for Lasso (using sklearn function)


r_lasso  = record_lasso(opt_lamb_l, threshs)
print r_lasso

