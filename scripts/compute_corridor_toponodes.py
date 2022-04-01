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
from skimage.morphology import disk,rectangle,square, binary_opening, binary_closing, remove_small_objects
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

from shapely.geometry import LineString
from shapely.affinity import scale

def calculate_line_angle(line): #list: [x0,y0,x1,y1]
	if (line[0]-line[2])==0:
		line_angle = 90
	else:
		line_angle = math.degrees(math.atan(float((line[1]-line[3]))/(line[0]-line[2])))
	return line_angle

def calculate_line_slope(line): #list: [x0,y0,x1,y1]
	if (line[0]-line[2])==0:
		line_slope = 999999
	else:
		line_slope = float((line[1]-line[3]))/(line[0]-line[2])
	return line_slope


#Parameters
#image_file_name = './data/greece_zoom20_23cmpix_binary.png'
image_file_name = '../data/input/ktima_Gerovasileioy_2020-07-21_Field_60_25cm_nonground_elevation_image_crop_only_rows.png'
compute_row_lines = True
show_plots = True
merge_row_lines = True
compute_toponodes_locations = True
save_output = True

# Read the image
img_bin_org = io.imread(image_file_name)
img_bin_org = rgb2gray(img_bin_org)
print('Original Dimensions : ',img_bin_org.shape) 

img_bin = img_bin_org.copy()
img_bin = img_bin > 0



if compute_row_lines:
	print "Computing vine row lines"

	# Divide binary image in clusters by means of connectivity rules
	img_labels = measure.label(img_bin,connectivity=2)
	num_of_clusters = np.max(img_labels)
	vine_rows = []
	angle_rows = []
	vine_rows_full_line = []

	print "num_of_clusters:", num_of_clusters

	clusters_to_test = [45,76]
	for current_cluster in range(1,num_of_clusters):#range(1,num_of_clusters):#clusters_to_test:#range(130,131):#num_of_clusters):
		#print "-------"
		print "cluster num: ", current_cluster

		# Compute lines for each cluster
		xx,yy = np.where(img_labels == current_cluster)
		img_cluster = np.zeros([img_bin.shape[0],img_bin.shape[1]])
		img_cluster[xx,yy] = 1 

		## hough transfom scikit-image
		tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 7200, endpoint=False)
		hspace, thetas, dists = hough_line(img_cluster, theta=tested_angles)
		_,best_angle_ind = np.where(hspace == np.max(hspace))
		#print "best angle ind",best_angle_ind
		#print "best angle ind len",len(best_angle_ind)
		#if len(best_angle_ind) < 3600:
			
		if len(best_angle_ind) > 1:
			index_to_pick = int(len(best_angle_ind)/2) #still don't know how to solve whcih of the max to choose. Put the mniddle but looking for a better solution
			best_angle_ind = best_angle_ind[index_to_pick] 


		# get other parallel lines
		intensities = hspace[:,best_angle_ind]
		best_distances_ind = argrelextrema(intensities, np.greater,order = 1)
		best_distances_ind = best_distances_ind[0]

		paralel_lines_distances = dists[best_distances_ind]
		paralel_lines_angle = thetas[best_angle_ind]

		#print "paralel lines angles:",paralel_lines_angle
		#print "paralel lines distances:",paralel_lines_distances

		x_max = img_cluster.shape[1]-1
		y_max = img_cluster.shape[0]-1


		for dist in paralel_lines_distances: #iterate over all paralel lines
			case = [False, False, False, False]
			# calculte the two points of the line crossing the outer limits
			# point 1
			#case 1
			point0_set = False
			point1_set = False

			x = 0
			y = (dist - x* np.cos(paralel_lines_angle))/np.sin(paralel_lines_angle)
			if not(x < 0 or x > x_max or y < 0 or y > y_max):
				case[0] = True
				if point0_set == False:
					x0 = x
					y0 = y
					point0_set = True

			#case 2
			y = 0
			x =(dist - y* np.sin(paralel_lines_angle))/np.cos(paralel_lines_angle)
			if not(x < 0 or x > x_max or y < 0 or y > y_max):
				case[1] = True
				if point0_set == False:
					x0 = x
					y0 = y
					point0_set = True
				else:
					x1 = x
					y1 = y

			#case 3
			x = x_max
			y = (dist - x* np.cos(paralel_lines_angle))/np.sin(paralel_lines_angle)
			if not(x < 0 or x > x_max or y < 0 or y > y_max):
				case[2] = True
				if point0_set == False:
					x0 = x
					y0 = y
					point0_set = True
				else:
					x1 = x
					y1 = y

			#case 4
			y = y_max
			x =(dist - y* np.sin(paralel_lines_angle))/np.cos(paralel_lines_angle)
			if not(x < 0 or x > x_max or y < 0 or y > y_max):
				case[3] = True
				if point0_set == False:
					x0 = x
					y0 = y
					point0_set = True
				else:
					x1 = x
					y1 = y


			x0 = int(round(x0))
			x1 = int(round(x1))
			y0 = int(round(y0))
			y1 = int(round(y1))

			#print "coordinates:", x0,y0,x1,y1

			rr,cc = line(y0,x0,y1,x1)
			values = img_cluster[rr,cc]

			# find where the lines ovelaps the cluster and find the two ending points of that overlapping
			cluster_line_indexes = np.where(values > 0)

			if np.size(cluster_line_indexes)>0:
				end_1 = [cc[cluster_line_indexes[0][0]],rr[cluster_line_indexes[0][0]]]
				end_2 = [cc[cluster_line_indexes[0][-1]],rr[cluster_line_indexes[0][-1]]]

				if end_1 != end_2:
					vine_rows_full_line.append([end_1[0],end_1[1],end_2[0],end_2[1]])


	# if show_plots:
		# ax[0].imshow(img_cluster, cmap=cm.gray)
		# ax[0].plot([x0,x1],[y0,y1])
		# ax[0].set_title('Detected lines')

		# ax[1].imshow(hspace)
		# ax[1].set_title('Hough transform')
		# ax[1].set_xlabel('Angles (degrees)')
		# ax[1].set_ylabel('Distance (pixels)')
		# ax[1].axis('image')

		# ax[2].plot(intensities)
		# ax[2].plot(best_distances_ind,intensities[best_distances_ind],'ro')

		# plt.tight_layout()
		# plt.show()

if merge_row_lines:
	print "Trying to merge rows"
	# take the vine rows detected and try to recronstrucnt gaps in the row
	radious_threshold = 5 #meters
	resolution = 4 #[pix/m]
	angle_threshold = 1 # degrees

	radius_threshold_pix = radious_threshold * resolution #pix
	print "radius threshold", radius_threshold_pix
	is_line_merged = True
	while is_line_merged:
		number_of_lines = len(vine_rows_full_line)
		merged_full_lines = []
		lines_to_detele = []

		is_line_merged = False
		# find end of rows points that are close to each other
		for org_line in range(0,number_of_lines):
			org_line_angle = calculate_line_angle(vine_rows_full_line[org_line][:])
			for dest_line in range(org_line+1,number_of_lines):
				#check the angle difference between the two lines
				dest_line_angle = calculate_line_angle(vine_rows_full_line[dest_line][:])
				if abs(org_line_angle-dest_line_angle) < angle_threshold:
					#if the angle is lower the the threshold check if any two ends are close enough
					#print "angle is lower"
					org_1 = vine_rows_full_line[org_line][0:2]
					org_2 = vine_rows_full_line[org_line][2:4]
					dest_1 = vine_rows_full_line[dest_line][0:2]
					dest_2 = vine_rows_full_line[dest_line][2:4]
	
					max_distance = 999999

					if distance.euclidean(org_1,dest_1) < radius_threshold_pix and distance.euclidean(org_1,dest_1) < max_distance:
						merged_line = [org_2[0],org_2[1],dest_2[0],dest_2[1]]
						merged_line_angle = calculate_line_angle(merged_line)
						if abs(org_line_angle-merged_line_angle) < angle_threshold and abs(dest_line_angle-merged_line_angle) < angle_threshold:
							#merged_full_lines.append(merged_line)
							line_to_merge = merged_line
							if not (org_line in lines_to_detele):
								lines_to_detele.append(org_line)
							if not (dest_line in lines_to_detele):
								lines_to_detele.append(dest_line)
							is_line_merged = True
							max_distance = distance.euclidean(org_1,dest_1)

					if distance.euclidean(org_1,dest_2) < radius_threshold_pix and distance.euclidean(org_1,dest_2) < max_distance:
						merged_line = [org_2[0],org_2[1],dest_1[0],dest_1[1]]
						merged_line_angle = calculate_line_angle(merged_line)
						if abs(org_line_angle-merged_line_angle) < angle_threshold and abs(dest_line_angle-merged_line_angle) < angle_threshold:
							#merged_full_lines.append(merged_line)
							line_to_merge = merged_line
							if not (org_line in lines_to_detele):
								lines_to_detele.append(org_line)
							if not (dest_line in lines_to_detele):
								lines_to_detele.append(dest_line)
							is_line_merged = True
							max_distance = distance.euclidean(org_1,dest_2)

					if distance.euclidean(org_2,dest_1) < radius_threshold_pix and distance.euclidean(org_2,dest_1) < max_distance:
						merged_line = [org_1[0],org_1[1],dest_2[0],dest_2[1]]
						merged_line_angle = calculate_line_angle(merged_line)
						if abs(org_line_angle-merged_line_angle) < angle_threshold and abs(dest_line_angle-merged_line_angle) < angle_threshold:
							#merged_full_lines.append(merged_line)
							line_to_merge = merged_line
							if not (org_line in lines_to_detele):
								lines_to_detele.append(org_line)
							if not (dest_line in lines_to_detele):
								lines_to_detele.append(dest_line)
							is_line_merged = True
							max_distance = distance.euclidean(org_2,dest_1)

					if distance.euclidean(org_2,dest_2) < radius_threshold_pix and distance.euclidean(org_2,dest_2) < max_distance:
						merged_line = [org_1[0],org_1[1],dest_1[0],dest_1[1]]
						merged_line_angle = calculate_line_angle(merged_line)
						if abs(org_line_angle-merged_line_angle) < angle_threshold and abs(dest_line_angle-merged_line_angle) < angle_threshold:
							#merged_full_lines.append(merged_line)
							line_to_merge = merged_line
							if not (org_line in lines_to_detele):
								lines_to_detele.append(org_line)
							if not (dest_line in lines_to_detele):
								lines_to_detele.append(dest_line)
							is_line_merged = True
							max_distance = distance.euclidean(org_2,dest_2)

				if is_line_merged:
					break
			if is_line_merged:
					break

		#delete short lines that have been merged and append the merged ones
		if is_line_merged:
			for d in sorted(lines_to_detele, reverse=True):
				del vine_rows_full_line[d]
			vine_rows_full_line.append(line_to_merge)

if compute_toponodes_locations:
 	#calculate the angles for all lines
 	#vine_rows_slope = []
 	avg_distance_between_rows = 2 # meters
 	resolution = 4 #pix/m
 	distance_between_nodes = 5 #meters

 	avg_distance_between_rows_pix = avg_distance_between_rows * resolution #pix
	distance_between_nodes_pix = distance_between_nodes * resolution # pix

 	# distance to pre-corridor nodes
 	distance_precorridor_nodes = 2
 	distance_precorridor_nodes_pix = distance_precorridor_nodes * resolution

	img_bin_add_row_area = img_bin.copy()
 	intra_corridor_topological_nodes = []
 	outer_corridor_topological_nodes = []

 	for row_number in range(0,len(vine_rows_full_line)):
 		vine_rows_slope = calculate_line_slope(vine_rows_full_line[row_number])
		a = vine_rows_full_line[row_number][0:2]
		b = vine_rows_full_line[row_number][2:4]
 		ab = LineString([a, b])
 		left = ab.parallel_offset(avg_distance_between_rows_pix/2, 'left')
		right = ab.parallel_offset(avg_distance_between_rows_pix/2, 'right')

		# left side
		p0_l = [left.boundary[0].x, left.boundary[0].y]
		p1_l = [left.boundary[1].x, left.boundary[1].y]
		row_length = distance.euclidean(p0_l,p1_l)
		actual_distance_between_nodes_pix = row_length/(np.ceil(row_length/distance_between_nodes_pix))
		number_of_divisions = row_length/actual_distance_between_nodes_pix
		x_increment = (p1_l[0]-p0_l[0])/number_of_divisions
		y_increment = (p1_l[1]-p0_l[1])/number_of_divisions
		left_points = []
		for p in range(0,(int(np.ceil(row_length/distance_between_nodes_pix))+1)):
			x = p0_l[0]+p*x_increment
			y = p0_l[1]+p*y_increment
			left_points.append([x,y])

		# right side
		p0_r = [right.boundary[0].x, right.boundary[0].y]
		p1_r = [right.boundary[1].x, right.boundary[1].y]
		row_length = distance.euclidean(p0_r,p1_r)
		actual_distance_between_nodes_pix = row_length/(np.ceil(row_length/distance_between_nodes_pix))
		number_of_divisions = row_length/actual_distance_between_nodes_pix
		x_increment = (p1_r[0]-p0_r[0])/number_of_divisions
		y_increment = (p1_r[1]-p0_r[1])/number_of_divisions
		right_points = []
		for p in range(0,(int(np.ceil(row_length/distance_between_nodes_pix))+1)):
			x = p0_r[0]+p*x_increment
			y = p0_r[1]+p*y_increment
			right_points.append([x,y])

		intra_corridor_topological_nodes.append([left_points,right_points])

		#calculating the pre corridor nodes
		extended_row_length = row_length + 2 * distance_precorridor_nodes_pix
		scaling_factor = extended_row_length/row_length
		e_left = scale(left,xfact= scaling_factor, yfact=scaling_factor, origin='center')
		e_right = scale(right,xfact= scaling_factor, yfact=scaling_factor, origin='center')
		p_l = [[e_left.boundary[0].x, e_left.boundary[0].y],[e_left.boundary[1].x, e_left.boundary[1].y]]
		p_r = [[e_right.boundary[0].x, e_right.boundary[0].y],[e_right.boundary[1].x, e_right.boundary[1].y]]

		outer_corridor_topological_nodes.append([p_l,p_r])

		# add thick line for each line - purpose: compute the navigation nodes in the remaining free space
 		left = ab.parallel_offset(avg_distance_between_rows_pix*0.6, 'left')
		right = ab.parallel_offset(avg_distance_between_rows_pix*0.6, 'right')	
		p0_l = [left.boundary[0].x, left.boundary[0].y]
		p1_l = [left.boundary[1].x, left.boundary[1].y]
		p0_r = [right.boundary[0].x, right.boundary[0].y]
		p1_r = [right.boundary[1].x, right.boundary[1].y]					
		rr, cc = polygon(np.array([p0_l[1],p1_l[1],p0_r[1],p1_r[1]]), np.array([p0_l[0],p1_l[0],p0_r[0],p1_r[0]]),img_bin_add_row_area.shape)
		img_bin_add_row_area[rr, cc] = 1

if show_plots:

	# fig, axes = plt.subplots(nrows=2, ncols=2)
	# axes[0,0].imshow(img_bin_org, cmap='gray')
	# axes[0,0].set_title('original')	

	# axes[0,1].imshow(img_bin, cmap='gray')
	# axes[0,1].set_title('grey')

	# axes[1,0].set_title('labels')
	# axes[1,0].imshow(img_labels, cmap='nipy_spectral')

	fig, axes = plt.subplots(nrows=1, ncols=1)
	axes.imshow(img_bin, cmap='gray')
	axes.set_title('row lines')	

	for line in range(0,len(vine_rows_full_line)):
		#plot lines
		axes.plot([vine_rows_full_line[line][0],vine_rows_full_line[line][2]],[vine_rows_full_line[line][1],vine_rows_full_line[line][3]], 'g', linewidth=2)	

		if compute_toponodes_locations:
			#plot topo nodes on each side of the line
			for side in range(0,2):
				for p in range(0,len(intra_corridor_topological_nodes[line][side])):
					axes.plot(intra_corridor_topological_nodes[line][side][p][0],intra_corridor_topological_nodes[line][side][p][1],'ro')
				for p in range(0,len(outer_corridor_topological_nodes[line][side])):	
					axes.plot(outer_corridor_topological_nodes[line][side][p][0],outer_corridor_topological_nodes[line][side][p][1],'yo')

	if compute_toponodes_locations:
		fig, axes = plt.subplots(nrows=1, ncols=1)
		axes.imshow(img_bin_add_row_area, cmap='gray')
		axes.set_title('row area')	
	plt.show()



if save_output:
	io.imsave(image_file_name[:-4]+"_added_offset.png",img_bin_add_row_area)