#!/usr/bin/env python

from scipy import ndimage
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from skimage import data, transform, io
from skimage.util import img_as_ubyte
from skimage.morphology import disk,rectangle,square, binary_opening, binary_closing, remove_small_objects
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


#Parameters
#image_file_name = './data/greece_zoom20_23cmpix_binary.png'
image_file_name = './data/gridmap2image.png'
apply_morphological_operations = True
compute_row_lines = True
show_plots = True

# Read the image
img_bin_org = io.imread(image_file_name)
img_bin_org = rgb2gray(img_bin_org)
print('Original Dimensions : ',img_bin_org.shape) 


if apply_morphological_operations:

	# Apply morphological operations
	print "Applying morphological operations"
	mask = square(3)
	img_bin_mod = img_bin_org > 0

	#img_bin_mod = binary_opening(img_bin_mod,mask)
	#img_bin_mod = binary_closing(img_bin_mod,mask)
	#img_bin_mod = remove_small_objects(img_bin_mod,min_size=30)
	#img_bin_mod = segmentation.clear_border(img_bin_mod)
	print "Done"


if compute_row_lines:
	print "Computing vine row lines"
	if show_plots:
		fig, axes = plt.subplots(1, 3, figsize=(15, 6))
		ax = axes.ravel()



	img_bin_mod = img_bin_org > 0

	# Divide binary image in clusters by means of connectivity rules
	img_labels = measure.label(img_bin_mod,connectivity=1)
	num_of_clusters = np.max(img_labels)
	vine_x = []
	vine_y = []

	print "num_of_clusters:", num_of_clusters

	for cluster_num in range(3130,3200):#num_of_clusters):
		print "-------"
		print "cluster num", cluster_num

		# Compute lines for each cluster
		#cluster_num =146
		xx,yy = np.where(img_labels == cluster_num)
		img_cluster = np.zeros([img_bin_org.shape[0],img_bin_org.shape[1]])
		img_cluster[xx,yy] = 1 



		#ax[0].imshow(img_cluster, cmap=cm.gray)
		#plt.show()


		## hough transfom scikit-image
		tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 3600, endpoint=False)
		hspace, thetas, dists = hough_line(img_cluster, theta=tested_angles)
		_,best_angle_ind = np.where(hspace == np.max(hspace))
		print "best angle ind",best_angle_ind
		print "best angle ind len",len(best_angle_ind)
		if len(best_angle_ind) < 3600:
			
			if len(best_angle_ind) > 1:
				best_angle_ind = best_angle_ind[0]


			# get other parallel lines
			intensities = hspace[:,best_angle_ind]
			best_distances_ind = argrelextrema(intensities, np.greater,order = 1)
			best_distances_ind = best_distances_ind[0]

			paralel_lines_distances = dists[best_distances_ind]
			paralel_lines_angle = thetas[best_angle_ind]

			print "paralel lines angles:",paralel_lines_angle
			print "paralel lines distances:",paralel_lines_distances

			for dist in paralel_lines_distances: #iterate over all paralel lines
				x0 = 0
				y0 = dist/np.sin(paralel_lines_angle)
				print img_cluster.shape[0]
				if y0 < 0 or y0 > img_cluster.shape[0]:
					y0 = 0
					x0 = dist/np.cos(paralel_lines_angle)

				x1 = img_cluster.shape[1]-1
				y1 = (dist - (img_cluster.shape[1]-1)* np.cos(paralel_lines_angle))/np.sin(paralel_lines_angle)
				if y1 < 0 or y1 > img_cluster.shape[0]:
					y1 = img_cluster.shape[0]-1
					x1 = (dist - img_cluster.shape[0]* np.sin(paralel_lines_angle))/np.cos(paralel_lines_angle)
				
				x0 = int(round(x0))
				x1 = int(round(x1))
				y0 = int(round(y0))
				y1 = int(round(y1))

				print "coordinates:", x0,y0,x1,y1

				rr,cc = line(y0,x0,y1,x1)
				try:
					values = img_cluster[rr,cc]

					cluster_line_indexes = np.where(values > 0)

					vine_x.append(cc[cluster_line_indexes[0][0]])
					vine_y.append(rr[cluster_line_indexes[0][0]])
					vine_x.append(cc[cluster_line_indexes[0][-1]])
					vine_y.append(rr[cluster_line_indexes[0][-1]])
				except:
					pass

				

	if show_plots:
		ax[0].imshow(img_cluster, cmap=cm.gray)
		ax[0].set_title('Detected lines')

		ax[1].imshow(hspace)
		ax[1].set_title('Hough transform')
		ax[1].set_xlabel('Angles (degrees)')
		ax[1].set_ylabel('Distance (pixels)')
		ax[1].axis('image')

		ax[2].plot(intensities)
		ax[2].plot(best_distances_ind,intensities[best_distances_ind],'ro')

		plt.tight_layout()
		plt.show()


if show_plots:

	fig, axes = plt.subplots(nrows=2, ncols=2)
	axes[0,0].imshow(img_bin_org, cmap='gray')
	axes[0,0].set_title('original')	

	axes[0,1].imshow(img_bin_mod, cmap='gray')
	axes[0,1].set_title('grey')

	axes[1,0].set_title('labels')
	axes[1,0].imshow(img_labels, cmap='nipy_spectral')

	fig, axes = plt.subplots(nrows=1, ncols=1)
	axes.imshow(img_bin_mod, cmap='gray')
	axes.set_title('row positions')	
	axes.plot(vine_x, vine_y, 'ro')
	plt.show()