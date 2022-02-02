#!/usr/bin/env python

from scipy import ndimage
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os.path
import time
import cv2
from skimage import data, transform, io
from skimage.util import img_as_ubyte
from skimage.morphology import disk,rectangle,square, binary_opening, binary_closing, remove_small_objects, remove_small_holes, skeletonize
from skimage.filters import rank
from skimage.color import rgb2gray
from skimage.draw import rectangle,polygon_perimeter,line,circle,polygon
from skimage import measure, segmentation
from skimage.transform import probabilistic_hough_line, hough_line, hough_line_peaks
from skimage.measure import LineModelND, ransac
from skimage.feature import peak_local_max
from skimage.filters import try_all_threshold
from skimage.color import rgb2hsv

from matplotlib import cm

from scipy import ndimage as ndi
from scipy.signal import argrelextrema
from scipy.spatial import distance
from scipy.stats import linregress
from scipy.spatial import Voronoi, voronoi_plot_2d

from shapely.geometry import LineString
from shapely.affinity import scale

import sknw #lib to build graph from skeleton
from networkx.algorithms.dag import transitive_closure

#Parameters
org_image_file_name = '../data/input/ktima_Gerovasileioy_2020-07-21_Field_60_25cm_nonground_elevation_image_morphological_operations.png'
rows_offset_image_file_name = '../data/input/ktima_Gerovasileioy_2020-07-21_Field_60_25cm_nonground_elevation_image_only_rows_added_offset.png'
show_plots = False
save_output = False
compute_voronoi = True

# Read the image
img_bin_org = io.imread(org_image_file_name)
print('Original Dimensions : ',img_bin_org.shape) 

img_bin_rows_offset = io.imread(rows_offset_image_file_name)

# merge both images
img_bin_merged = img_bin_org + img_bin_rows_offset
img_bin_merged = img_bin_merged > 0
img_bin_merged = remove_small_holes(img_bin_merged)


if compute_voronoi:
	#calculate all points that are obstacles
	xx,yy = np.where(img_bin_merged > 0)
	obstacle_pix = []
	for i in range(0, len(xx)):
		obstacle_pix.append([yy[i],xx[i]])

	#vor = Voronoi(obstacle_pix)
	#fig = voronoi_plot_2d(vor,show_vertices = False)
	#plt.show()
	#possible_nav_toponodes = vor.vertices
	img_bin_merged_oppo = img_bin_merged == 0
	edt_img = ndimage.distance_transform_edt(img_bin_merged_oppo)

	skeleton = skeletonize(img_bin_merged_oppo)

	# build graph from skeleton
	graph = sknw.build_sknw(skeleton, iso=False, ring=False, full=True)

	# outdeg = graph.out_degree()
	# to_remove = [n for n in outdeg if outdeg[n] == 1]
	# graph.remove_nodes_from(to_remove)

	#print graph.nodes[1]
	print "number of nodes"
	print graph.number_of_nodes()

	node_removed = True
	while node_removed == True:
		node_removed = False
		nodes_to_remove = []
		for i in graph.nodes():
		 	if graph.degree(i) == 1:
		 		nodes_to_remove.append(i)
		 		node_removed = True
		graph.remove_nodes_from(nodes_to_remove)


show_plots = True
if show_plots:

	fig, axes = plt.subplots(nrows=1, ncols=2)

	# axes[0].imshow(img_bin_org, cmap='gray')
	# axes[0].set_title('original')	

	# axes[1].imshow(img_bin_rows_offset, cmap='gray')
	# axes[1].set_title('rows offset')

	axes[0].set_title('merging')
	axes[0].imshow(img_bin_merged, cmap='gray')

	axes[1].set_title('edt')
	axes[1].imshow(img_bin_merged, cmap='gray')

	#draw edges by pts
	for (s,e) in graph.edges():
	    ps = graph[s][e]['pts']
	    axes[1].plot(ps[:,1], ps[:,0], 'green')

	#nodes = graph.nodes()
	nodes = graph.nodes()
	ps = np.array([nodes[i]['o'] for i in nodes])
	axes[1].plot(ps[:,1], ps[:,0], 'r.')

	plt.show()

# if save_output:
# 	io.imsave(image_file_name[:-4]+"_added_offset.png",img_bin_add_row_area)