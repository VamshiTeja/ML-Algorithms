# -*- coding: utf-8 -*-
# @Author: vamshi
# @Date:   2018-09-23 23:38:51
# @Last Modified by:   vamshi
# @Last Modified time: 2018-09-24 14:16:30

import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import time
from matplotlib.patches import Ellipse
from sklearn.cluster import KMeans


class GMM:

	def __init__(self,X,K=3,max_iters=100,eps=1e-4):
		'''
			X: nxd matrix
			K : Number of clusters
		'''
		self.K = K
		self.X = X
		self.max_iters = max_iters
		self.eps = eps

		self.d = X.shape[1]
		self.n = X.shape[0]
		self.likelihood = 0

		self._parameter_init_()


	def _parameter_init_(self):
		#uniform mixing initially
		self.w = np.array([1./self.K]*self.K)

		#KMeans init for means
		kmeans = KMeans(n_clusters=self.K, random_state=0).fit(self.X)
		self.mu = kmeans.cluster_centers_

		#identity covariance initially
		sigma = []
		for i in range(self.K):
			sigma.append(np.identity(self.d))
		self.sigma = np.array(sigma)


	def calc_posterior(self):
		'''
			output: pos-nxk matrix 
		'''
		r = np.zeros(shape=(self.n,self.K))
		dist = []
		for i in range(self.K):
			dist.append(multivariate_normal.pdf(self.X,self.mu[i],self.sigma[i]))

		dist = np.array(dist).T
		pos = []
		for i in range(self.n):
			p = np.multiply(dist[i],self.w)
			pos.append(p/np.sum(p))

		pos = np.array(pos)
		return pos


	def _update_parameters(self,pos):
		#update mu
		new_mu = np.zeros(shape=(self.K,self.d))
		for i in range(self.K):
			pos_i = pos[:,i]
			new_mu[i,:] = np.dot(pos_i, self.X)/np.sum(pos_i)
		#update sigma
		new_sigma = []
		for i in range(self.K):
			a = np.zeros(shape=(self.d,self.d))
			for j in range(self.n):
				c = np.array(self.X[j]-self.mu[i])
				c = np.expand_dims(c, axis=1)
				a += pos[j][i]*np.matmul(c, c.T)
			new_sigma.append(a/np.sum(pos[:,i]))

		new_sigma = np.array(new_sigma)
	
		#update w
		new_w = []
		new_w = np.sum(pos,axis=0)/self.n
		# for i in range(self.K):
		# 	new_w.append(np.sum(self.pos[:,i])/self.n)

		return new_mu, new_w, new_sigma


	def _compute_likelihood(self,mu,sigma,w):
		dist = []
		for i in range(self.K):
			dist.append(multivariate_normal.pdf(self.X,mu[i],sigma[i]))

		dist = np.array(dist).T		

		likelihood = 0
		for i in range(self.n):
			likelihood += np.log(np.matmul(w,dist[i]))
		return likelihood

	def _iter_update(self):
		pos = self.calc_posterior()
		new_mu, new_w, new_sigma = self._update_parameters(pos)
		likelihood = self._compute_likelihood(new_mu, new_sigma, new_w)
		self.mu, self.sigma, self.w = new_mu, new_sigma, new_w
		likelihood_error = np.mean(np.abs(likelihood-self.likelihood))
		self.likelihood = likelihood

		return self.mu, self.sigma, self.w, pos, likelihood, likelihood_error

	def _run_(self):

		iterations = 0
		while(iterations<self.max_iters):
			self.mu, self.sigma, self.w, pos, likelihood, err = self._iter_update()
			cluster_assign = np.argmax(pos,axis=1)
			
			if(err<=self.eps):
				return self.mu, self.sigma, self.w, cluster_assign

			iterations += 1

		print("Not converged. Try Increasing max iters. Returning current state")
		return self.mu, self.sigma, self.w, cluster_assign


###################################################################
#helper functions for plotting

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))
        
def plot_gmm(means, covars, weights, X, labels, ax=None):
    ax = ax or plt.gca()
    
    ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    ax.axis('equal')
    
    w_factor = 0.2 / np.max(weights)
    for pos, covar, w in zip(means, covars, weights):
        draw_ellipse(pos, covar, alpha=w * w_factor)


###################################################################
#K=2
print("Test for 2 clusters")


mean1 = [1,3]
cov1  = [[1,2],[2,1]]
X1 = np.random.multivariate_normal(mean=mean1, cov=cov1, size=(100))
y1 = np.zeros(shape=(100,))

mean2 = [4,1]
cov2  = [[4,-6],[-6,4]]
X2 = np.random.multivariate_normal(mean=mean2, cov=cov2, size=(100))
y2 = np.zeros(shape=(100,))+1

X = np.concatenate((X1, X2), axis=0)
y = np.concatenate((y1, y2))

plt.figure(0)
plt.scatter(X[:,0], X[:,1],c=y,s=50,label="original data")
plt.savefig("2_orig.png")
plt.legend(loc=0)

model = GMM(X, K=2, max_iters=100)
mu, sigma, w , cluster_assign = model._run_()

print("Obtained Parameters are:")
print("Mean:")
print(mu)
print("\nCovariances:")
print(sigma)
print("\nMixing Parameter")
print(w)
print("\nFinal Cluster Assignments are:")
print(cluster_assign)

plt.figure(1)
plt.scatter(X[:,0], X[:,1],c=cluster_assign,s=50,label="custered data")
pl = plt.scatter(mu[:,0], mu[:,1],marker='*',color='r')
plt.legend(loc=0)
plt.savefig("2_clusters.png")

plt.figure(2)
plot_gmm(mu, sigma, w, X, cluster_assign)
plt.savefig("2_cluster_shapes.png")
plt.show()

###################################################################
#K=3
print("Test for 3 clusters")

mean1 = [1,3]
cov1  = [[1,2],[2,1]]
X1 = np.random.multivariate_normal(mean=mean1, cov=cov1, size=(100))
y1 = np.zeros(shape=(100,))

mean2 = [4,1]
cov2  = [[2,-3],[-3,2]]
X2 = np.random.multivariate_normal(mean=mean2, cov=cov2, size=(150))
y2 = np.zeros(shape=(150,))+1

mean3 = [6,8]
cov3  = [[2,-3],[-3,2]]
X3 = np.random.multivariate_normal(mean=mean3, cov=cov3, size=(200))
y3 = np.zeros(shape=(200,))+2

X = np.concatenate((X1, X2,X3), axis=0)
y = np.concatenate((y1, y2,y3))

plt.figure(0)
plt.scatter(X[:,0], X[:,1],c=y,s=50,label="original data")
plt.savefig("3_orig.png")
plt.legend(loc=0)

model = GMM(X, K=3, max_iters=100)
mu, sigma, w , cluster_assign = model._run_()

print("\n\nObtained Parameters are:")
print("Mean:")
print(mu)
print("\nCovariances:")
print(sigma)
print("\nMixing Parameter")
print(w)
print("\nFinal Cluster Assignments are:")
print(cluster_assign)

plt.figure(1)
plt.scatter(X[:,0], X[:,1],c=cluster_assign,s=50,label="custered data")
pl = plt.scatter(mu[:,0], mu[:,1],marker='*',color='r')
plt.savefig("3_clusters.png")
plt.legend(loc=0)

plt.figure(2)
plot_gmm(mu, sigma, w, X, cluster_assign)
plt.savefig("3_cluster_shapes.png")
plt.show()

###################################################################
#K=4
print("Test for 4 clusters")
mean1 = [1,3]
cov1  = [[1,2],[2,1]]
X1 = np.random.multivariate_normal(mean=mean1, cov=cov1, size=(50))
y1 = np.zeros(shape=(50,))

mean2 = [4,1]
cov2  = [[2,-3],[-3,2]]
X2 = np.random.multivariate_normal(mean=mean2, cov=cov2, size=(100))
y2 = np.zeros(shape=(100,))+1

mean3 = [6,8]
cov3  = [[1,-3],[-4,2]]
X3 = np.random.multivariate_normal(mean=mean3, cov=cov3, size=(200))
y3 = np.zeros(shape=(200,))+2

mean4 = [9,3]
cov4  = [[4,2],[-5,2]]
X4 = np.random.multivariate_normal(mean=mean4, cov=cov4, size=(50))
y4 = np.zeros(shape=(50,))+3

X = np.concatenate((X1, X2,X3,X4), axis=0)
y = np.concatenate((y1, y2,y3,y4))

plt.figure(0)
plt.scatter(X[:,0], X[:,1],c=y,s=50,label="original data")
plt.savefig("4_orig.png")
plt.legend(loc=0)

model = GMM(X, K=4, max_iters=100)
mu, sigma, w , cluster_assign = model._run_()

print("\n\nObtained Parameters are:")
print("Mean:")
print(mu)
print("\nCovariances:")
print(sigma)
print("\nMixing Parameter")
print(w)
print("\nFinal Cluster Assignments are:")
print(cluster_assign)

plt.figure(1)
plt.scatter(X[:,0], X[:,1],c=cluster_assign,s=50,label="custered data")
pl = plt.scatter(mu[:,0], mu[:,1],marker='*',color='r')
plt.savefig("4_clusters.png")
plt.legend(loc=0)

plt.figure(2)
plot_gmm(mu, sigma, w, X, cluster_assign)
plt.savefig("4_cluster_shapes.png")
plt.show()

###################################################################
