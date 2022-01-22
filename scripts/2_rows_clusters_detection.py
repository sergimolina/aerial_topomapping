#!/usr/bin/env python

from scipy import ndimage
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from skimage import data, transform, io
from skimage.util import img_as_ubyte
from skimage.morphology import disk,rectangle,square, binary_opening, binary_closing, remove_small_objects, remove_small_holes, binary_erosion, thin
from skimage.filters import rank
from skimage.color import rgb2gray
from skimage.draw import rectangle,polygon_perimeter,line,circle
import os.path
import time
from skimage import measure, segmentation
from skimage.transform import probabilistic_hough_line, hough_line, hough_line_peaks
from skimage.measure import LineModelND, ransac
from matplotlib import cm
import cv2
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from scipy.signal import argrelextrema
from skimage.filters import try_all_threshold
from skimage.color import rgb2hsv
from sklearn.decomposition import PCA

#Parameters
#image_file_name = '../data/greece_zoom20_23cmpix_binary.png'
image_file_name = '../data/input/ktima_Gerovasileioy_2020-07-21_Field_60_25cm_nonground_elevation_image.png'
apply_morphological_operations = True
apply_row_cluster_detection = True
show_plots = True
save_output = True

# Read the image
img_org = io.imread(image_file_name)
img_grey = rgb2gray(img_org)


# binarization of the grey scale elevation image
img_bin = img_grey > 0
img_bin_org = img_bin
print('Original Dimensions : ',img_bin.shape) 


if apply_morphological_operations:

	# Apply morphological operations
	print "-- Applying morphological operations --"
	# mask = circle(1,1,1)
	
	# img_bin = binary_opening(img_bin,mask)
	# img_bin = binary_closing(img_bin,mask)
	img_bin = remove_small_objects(img_bin,min_size=30)
	img_bin = remove_small_holes(img_bin)
	img_bin = segmentation.clear_border(img_bin)
	# img_bin = thin(img_bin,1)

	img_morph = img_bin
	print "Done"


if apply_row_cluster_detection:

	print "-- Applying cluster classification --"

	# Divide binary image in clusters by means of connectivity rules
	img_labels = measure.label(img_bin,connectivity=2)
	num_of_clusters = np.max(img_labels)
	print "Num_of_clusters before classification:", num_of_clusters

	img_only_rows = np.zeros([img_bin_org.shape[0],img_bin_org.shape[1]])
	number_of_clusters_after = 0
	ratios = np.array([])

	for cluster_num in range(1,num_of_clusters):#num_of_clusters):

		# find all the x and y points beloging to the same cluster
		xx,yy = np.where(img_labels == cluster_num)
		X = np.array([xx,yy])
		X = X.T

		# compute PCA over the 2d features to obtain the eigen vectors describing the cluster
		pca = PCA(n_components=2).fit(X)
		ratio = pca.explained_variance_ratio_[0]/pca.explained_variance_ratio_[1]
		ratios = np.append(ratios,ratio)
		if ratio > 30:
		 	img_only_rows[xx,yy] = 1 
		 	number_of_clusters_after = number_of_clusters_after + 1

	print "Num_of_clusters after classification:", number_of_clusters_after
	print "Done"


if show_plots:

	fig, axes = plt.subplots(nrows=1, ncols=3)

	axes[0].imshow(img_bin_org, cmap='gray')
	axes[0].set_title('Original Binary')

	if apply_morphological_operations:
		axes[1].imshow(img_morph, cmap='gray')
		axes[1].set_title('Morphological Operations')

	if apply_row_cluster_detection:
		axes[2].imshow(img_only_rows, cmap='gray')
		axes[2].set_title('Only rows')


	plt.show()

if save_output:
	io.imsave(image_file_name[:-4]+"_only_rows.png",img_only_rows)