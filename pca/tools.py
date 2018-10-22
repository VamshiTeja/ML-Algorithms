
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


def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  if (images.shape[3] in (3,4)):
    c = images.shape[3]
    img = np.zeros((h * size[0], w * size[1], c))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w, :] = image
    return img
  elif images.shape[3]==1:
    img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
    return img
  else:
    raise ValueError('in merge(images,size) images parameter '
                     'must have dimensions: HxW or HxWx3 or HxWx4')

def imsave(images, size, path):
  image = np.squeeze(merge(images, size))
  return scipy.misc.imsave(path, image)

def inverse_transform(images):
  return images

def save_images(images, size, image_path):
  return imsave(inverse_transform(images), size, image_path)

def image_manifold_size(num_images):
  manifold_h = int(np.floor(np.sqrt(num_images)))
  manifold_w = int(np.ceil(np.sqrt(num_images)))
  assert manifold_h * manifold_w == num_images
  return manifold_h, manifold_w

def extract_patches(img,kernel=10):
	patches = []
	h,w = img.shape[0], img.shape[1]
	num_patches = ((h-kernel)/kernel +1)
	for i in range(num_patches):
		for j in range(num_patches):
			patches.append(img[i*kernel:(i+1)*kernel,j*kernel:(j+1)*kernel]) 
	return np.array(patches)

def to_img(patches, kernel=10):
	num_patches = patches.shape[0]
	img = np.zeros(shape=(200,200))
	for i in range(int(np.sqrt(num_patches))):
		for j in range(int(np.sqrt(num_patches))):
			img[i*kernel:(i+1)*kernel,j*kernel:(j+1)*kernel] = patches[i*int(np.sqrt(num_patches))+j]
	return img