# -*- coding: utf-8 -*-
# @Author: vamshi
# @Date:   2018-09-10 09:23:55
# @Last Modified by:   vamshi
# @Last Modified time: 2018-09-16 21:46:50

import os,sys
import numpy as np
import sklearn
import pandas
import matplotlib.pyplot as plt   
import cv2

#*******************************************************************************#

class KMeans:
	def __init__(self, X, K=3, eps=1e-4, iters=100):
		'''
			X: nxd matrix
			K: Number of clusters
			eps: Error tolerance for convergence of centroids
		'''
		self.X = X
		self.K = K
		self.eps = eps

		self.n = self.X.shape[0]
		self.d = self.X.shape[1]

		self.centroids = self.centroid_init()
		self.max_iters = iters


	def centroid_init(self):
		ind = np.random.choice(range(self.n), size=self.K)
		centroids = self.X[ind,:]
		print("Initial Centroids:  ") 
		print(centroids)
		return centroids

	def dist(self,x,y,axis=0):
		return np.sqrt(np.sum((x-y)**2,axis=axis))

	def dist_bw_Xncluster(self,X,centroids):
		dist = np.zeros(shape=(self.n,self.K))
		for i in range(self.n):
			for j in range(self.K):
				dist[i][j] = self.dist(X[i], centroids[j])
		return dist

	def compute_new_centroids(self,cluster_assign):
		new_centers = []
		for i in range(self.K):
			ind = np.where(cluster_assign==i)
			x_ind = self.X[ind]
			center_i = np.mean(x_ind,axis=0)
			new_centers.append(center_i)
		return np.array(new_centers)

	def iter_update(self):

		dist = self.dist_bw_Xncluster(self.X, self.centroids)
		#get new cluster assignments
		new_cluster_assign = np.argmin(dist,axis=1)
		dist_bw_cluster = np.mean(np.min(dist,axis=1))
		#compute new centroids
		new_centroids = self.compute_new_centroids(new_cluster_assign)
		error = self.dist(self.centroids, new_centroids,axis=1)
		self.centroids = new_centroids
		return new_centroids, error, new_cluster_assign, dist_bw_cluster


	def run(self):
		it = 0
		while(it<self.max_iters):
			new_centroids, error, new_cluster_assign, dist_bw_xn_cluster = self.iter_update()
			print("centroids at step %d: "%it)
			print(new_centroids)
			print("Error(distance between cluster assignments and centroids) at step %d : %f"%(it,dist_bw_xn_cluster))
			print("Error(distance between current and previous clusters) at step %d : %f"%(it,np.mean(error)))
			if(error.any()<self.eps):
				return new_centroids, error, new_cluster_assign
			it = it + 1
		print("Not converged. Try Increasing max iters. Returning current state")
		return self.centroids, error, new_cluster_assign

#*******************************************************************************#
#On Gaussian
X1 = np.random.rand(50, 2) + 5
y1 = np.zeros(shape=(50,))
X2 = 2*np.random.rand(50, 2) + 2
y2 = np.ones(shape=(50,))
X3 = 2.5*np.random.rand(50, 2) + 3
y3 = np.zeros(shape=(50,))+2

X = np.concatenate((X1, X2,X3), axis=0)
y = np.concatenate((y1, y2,y3))

k = 3
model = KMeans(X,K=k,eps=0.1)
centroids, _, cluster_assign = model.run()
print("Final cluster assignments are:")
print(cluster_assign)
plt.scatter(X[:,0], X[:,1],c=cluster_assign,s=50)
pt = plt.scatter(centroids[:,0], centroids[:,1],label="Centroids")
plt.legend(loc=0)
plt.show()
#*******************************************************************************#
#On Image
img = cv2.imread("img.jpeg",1)
h,w,c = img.shape[0], img.shape[1], img.shape[2]
f = np.reshape(img, newshape=(h*w,c))

k = 10
model = KMeans(f,K=k,eps=0.1)
centroids, _, cluster_assign = model.run()
print("Final cluster assignments are:")
print(cluster_assign)

cluster_img = np.reshape(cluster_assign, newshape=(h,w))
cluster_img = cluster_img*255/(k-1)
cv2.imwrite("cluster_%d.png"%k,cluster_img)


