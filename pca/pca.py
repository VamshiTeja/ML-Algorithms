# -*- coding: utf-8 -*-
# @Author: vamshi
# @Date:   2017-11-10 10:53:09
# @Last Modified by:   vamshi
# @Last Modified time: 2018-09-15 17:00:04
#PCA Implementation

import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn import datasets 
from sklearn.decomposition import PCA as pca
from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
import pylab as pl
import scipy
import cv2
from tools import *


#*******************************************************************************#

class PCA:

	def cov_matrix(self,X):
		return np.cov(X)

	def get_eigen(self,mat):
		eigen_val, eigen_vec = np.linalg.eig(mat)
		return eigen_val,eigen_vec

	def PCA(self,X,num_top_eigen_val,image=0):
		'''
		 	X: Input data must be of the form num_dim x num_points
		 	num_top_eigen_val: number of top eigen values to consider

		 	returns num_points x num_top_eigen_val transformed matrix
		'''
		#Finds covariance matrix of input data after normalising
		mean = np.mean(X,axis=0)
		X = (X-mean)
		#var  = np.var(X,axis=0)
		
		cov = self.cov_matrix(X)
		#Extracts eigen values and eigen vectors from covariance matrix
		eigen_val, eigen_vec = self.get_eigen(cov)
		eig_pairs = [(np.abs(eigen_val[i]),eigen_vec[:,i]) for i in range(len(eigen_val))]
		#sort eigen vectors based on eigen values
		eig_pairs.sort(cmp,key=lambda x: x[0], reverse=True)
		W = []
		for i in range(num_top_eigen_val):
			W.append(eig_pairs[i][1])
		W = np.vstack(W)

		#Transform based on top eigen values
		transformed = np.dot(W, X)
		return transformed.T,eigen_vec, W

#*******************************************************************************#
#On Iris Dataset

data_dir = "./Iris.csv"
headers = ['Id','SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species']
np.random.seed(42)
def read_data(data_dir,headers):
	df = pd.read_csv(data_dir,header=0,names=headers)
	df[headers[5]] = df[headers[5]].astype('category')
	df[headers[5]] = df[headers[5]].cat.codes
	mat = df.as_matrix()
	X = mat[:,1:5]
	y = mat[:,5]
	return X,y

X_iris,y_iris = read_data(data_dir,headers)
#pca = pca(n_components=2)
#X_pca = pca.fit_transform(X_iris)
#X_pca = PCA(X_iris)
X_pca_model = PCA()
X_pca,_,_ = X_pca_model.PCA(X_iris.T,2)

plt.figure(1)
plt.scatter(X_pca[:,0],X_pca[:,1],c=y_iris)
plt.legend(('a','b','c'))
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title("PCA for Iris Dataset")

X_tsne = TSNE(n_components=2,random_state=42).fit_transform(X_iris)

plt.figure(2)
plt.scatter(X_tsne[:,0], X_tsne[:,1],c=y_iris)
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title("TSNE for Iris Dataset")
plt.show()


#*******************************************************************************#

#On Peoples Dataset

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape 

# for machine learning we use the 2 data directly (as relative pixel positions info is ignored by this model)
X = lfw_people.data
n_features = X.shape[1]

X_imgs = lfw_people.images[0:64]
X_imgs = np.expand_dims(X_imgs, axis=-1)
print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print ("height: %d, width: %d"%(h,w))

n_components = 100

model = PCA()
X_pca,eigen_vec, W = model.PCA(X.T,n_components)

X_proj = np.dot(X_pca,W)
X_proj = np.reshape(X_proj, newshape=[-1,h,w,1])[0:64,:]

eigen_vec_top = eigen_vec.T[0:n_components,:]
eigen_vec_top = np.reshape(eigen_vec_top, newshape=[n_components,h,w,1])
#save eigen vectors
save_images(eigen_vec_top, image_manifold_size(eigen_vec_top.shape[0]), image_path="./eigen_faces.png")
save_images(X_proj, image_manifold_size(X_proj.shape[0]), image_path="./faces_projected.png")
save_images(X_imgs[0:64], image_manifold_size(X_imgs[0:64].shape[0]), image_path="./faces.png")
#*******************************************************************************#
#PCA generally wont work when two variance directions are not orthogonal
#It considers all basis to be orthogonal

#examples of PCA failing, multivariate gaussian
#gaussian 
X1 = np.random.multivariate_normal(mean=(1,2), cov=[[2,3],[3,4]],size=(500))
y1 = np.zeros(shape=(500,),dtype=np.int)
X2 = np.random.multivariate_normal(mean=(0,1), cov=[[1,-2],[-2,3]],size=(500))
y2 = np.ones(shape=(500,),dtype=np.int)

X = np.concatenate((X1, X2), axis=0)
y = np.concatenate((y1, y2))

plt.figure(1)
plt.scatter(X[:,0], X[:,1],c=y,s=50)

model = PCA()
X_pca,_,_ = model.PCA(X.T,2)

plt.figure(2)
plt.scatter(X_pca[:,0], X_pca[:,1],c=y,s=50)
plt.show()

#*******************************************************************************#

#examples of PCA failing, on swiss roll dataset
# generate the swiss roll
n_samples, n_features = 500, 3
n_turns, radius = 1.2, 1.0
rng = np.random.RandomState(0)
t = rng.uniform(low=0, high=1, size=n_samples)
data = np.zeros((n_samples, n_features))

# generate the 2D spiral data driven by a 1d parameter t
max_rot = n_turns * 2 * np.pi
data[:, 0] = radius = t * np.cos(t * max_rot)
data[:, 1] = radius = t * np.sin(t * max_rot)
data[:, 2] = rng.uniform(-1, 1.0, n_samples)
manifold = np.vstack((t * 2 - 1, data[:, 2])).T.copy()
colors = manifold[:, 0]

model = PCA()
X_swiss_pca,_,_ = model.PCA(data.T,num_top_eigen_val=2)
X_swiss_tsne = TSNE(n_components=2).fit_transform(data)

plt.figure(3)
plt.scatter(X_swiss_pca[:,0],X_swiss_pca[:,1], c=colors,cmap=plt.cm.Spectral)
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title("PCA for Swiss Roll Dataset")

plt.figure(4)
plt.scatter(X_swiss_tsne[:,0],X_swiss_tsne[:,1], c=colors,cmap=plt.cm.Spectral)
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title("TSNE for Swiss Roll Dataset")
plt.show()
