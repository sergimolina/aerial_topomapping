from scipy import ndimage
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os.path
import time
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

import rasterio
import aerial_topomapping as at

def compute_voronoi(binary_image):
	print("-- Computing voronoi of the free space --")
	xx,yy = np.where(binary_image > 0)
	obstacle_pix = []
	for i in range(0, len(xx)):
		obstacle_pix.append([yy[i],xx[i]])

	img_bin_merged_oppo = binary_image == 0
	edt_img = ndimage.distance_transform_edt(img_bin_merged_oppo)

	skeleton = skeletonize(img_bin_merged_oppo)

	# build graph from skeleton
	graph = sknw.build_sknw(skeleton, iso=False, ring=False, full=True)

	#print graph.nodes[1]
	print("number of nodes")
	print(graph.number_of_nodes())

	# remove nodes with only 1 edge
	node_removed = True
	while node_removed == True:
		node_removed = False
		nodes_to_remove = []
		for i in graph.nodes():
			if graph.degree(i) == 1:
				nodes_to_remove.append(i)
				node_removed = True
		graph.remove_nodes_from(nodes_to_remove)

	print("-- Done --")
	return graph

def create_topogrid(binary_image, grid_size):
	# check the size of the image
	length,width = np.shape(binary_image)
	nodes = []
	for c in range(grid_size,width, grid_size):
		for r in range(grid_size,length,grid_size):
			if binary_image[r,c] == 0:
				nodes.append([r,c])
	
	return nodes

# Read the image
image_filename = '../../data/riseholme_correction/riseholme_correction.tif'
show_plots = True

image = rasterio.open(image_filename)
band1 = image.read(1)
band1_mod = at.apply_binarisation(band1)

graph = compute_voronoi(band1_mod)

nodes_grid = create_topogrid(band1_mod, 50)

if show_plots:

	fig, axes = plt.subplots(nrows=1, ncols=1)

	axes.imshow(band1, cmap='gray')

	for n in nodes_grid:
		axes.plot(n[1], n[0], 'y.')

	# #draw edges by pts
	for (s,e) in graph.edges():
		if len(graph)> 1:
			ps = graph[s][e]['pts']
			axes.plot(ps[:,1], ps[:,0], 'green')

	nodes = graph.nodes()
	ps = np.array([nodes[i]['o'] for i in nodes])
	axes.plot(ps[:,1], ps[:,0], 'r.')

	plt.show()
