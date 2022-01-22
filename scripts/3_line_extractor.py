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
from skimage.draw import rectangle,polygon_perimeter,line,circle
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


def calculate_line_angle(line): #list: [x0,y0,x1,y1]
	if (line[0]-line[2])==0:
		line_angle = 90
	else:
		line_angle = math.degrees(math.atan(float((line[1]-line[3]))/(line[0]-line[2])))
	return line_angle

#Parameters
#image_file_name = './data/greece_zoom20_23cmpix_binary.png'
image_file_name = '../data/input/ktima_Gerovasileioy_2020-07-21_Field_60_25cm_nonground_elevation_image_only_rows.png'
compute_row_lines = True
show_plots = True
merge_row_lines = True
compute_toponodes_locations = False

# Read the image
img_bin_org = io.imread(image_file_name)
img_bin_org = rgb2gray(img_bin_org)
print('Original Dimensions : ',img_bin_org.shape) 

img_bin = img_bin_org

if compute_row_lines==True:
	print "Computing vine row lines"

	# Divide binary image in clusters by means of connectivity rules
	img_labels = measure.label(img_bin,connectivity=2)
	num_of_clusters = np.max(img_labels)
	vine_rows = []
	angle_rows = []
	vine_rows_full_line = []
	angle_rows_full_line = []

	print "num_of_clusters:", num_of_clusters

	clusters_to_test = [45,76]
	for current_cluster in range(1,num_of_clusters):#clusters_to_test:#range(130,131):#num_of_clusters):
		#print "-------"
		print "cluster num", current_cluster

		# Compute lines for each cluster
		xx,yy = np.where(img_labels == current_cluster)
		img_cluster = np.zeros([img_bin_org.shape[0],img_bin_org.shape[1]])
		img_cluster[xx,yy] = 1 

		## hough transfom scikit-image
		tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 7200, endpoint=False)
		hspace, thetas, dists = hough_line(img_cluster, theta=tested_angles)
		_,best_angle_ind = np.where(hspace == np.max(hspace))
		#print "best angle ind",best_angle_ind
		#print "best angle ind len",len(best_angle_ind)
		#if len(best_angle_ind) < 3600:
			
		if len(best_angle_ind) > 1:
			best_angle_ind = best_angle_ind[0] #still don't know how to solve whcih of the max to choose


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
			x0 = 0
			y0 = dist/np.sin(paralel_lines_angle)
			if y0 < 0 or y0 > y_max:		
				x0 = dist/np.cos(paralel_lines_angle)
				y0 = 0

			x1 = x_max
			y1 = (dist - (x_max)* np.cos(paralel_lines_angle))/np.sin(paralel_lines_angle)
			if y1 < 0 or y1 > y_max:	
				x1 = (dist - x_max* np.sin(paralel_lines_angle))/np.cos(paralel_lines_angle)
				y1 = y_max

			if x1 < 0 or x1 > x_max:		
			 	x1 = dist/np.cos(paralel_lines_angle)
			 	y1 = 0	

			x0 = int(round(x0))
			x1 = int(round(x1))
			y0 = int(round(y0))
			y1 = int(round(y1))

			#print "coordinates:", x0,y0,x1,y1
			rr,cc = line(y0,x0,y1,x1)
			values = img_cluster[rr,cc]

			cluster_line_indexes = np.where(values > 0)

			if np.size(cluster_line_indexes)>0:
				end_1 = [cc[cluster_line_indexes[0][0]],rr[cluster_line_indexes[0][0]]]
				end_2 = [cc[cluster_line_indexes[0][-1]],rr[cluster_line_indexes[0][-1]]]

				# vine_rows.append(end_1)
				# vine_rows.append(end_2)
				# if (end_1[0]-end_2[0])==0:
				# 	row_angle = 90
				# else:
				# 	row_angle = math.degrees(math.atan(float((end_1[1]-end_2[1]))/(end_1[0]-end_2[0])))

				# angle_rows.append(row_angle)
				# angle_rows.append(row_angle)

				vine_rows_full_line.append([end_1[0],end_1[1],end_2[0],end_2[1]])
				angle_rows_full_line.append(calculate_line_angle([end_1[0],end_1[1],end_2[0],end_2[1]]))

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

# if merge_row_lines:
# 	print "Trying to merge rows"
# 	# take the vine rows detected and try to recronstrucnt gaps in the row
# 	radious_threshold = 10 #meters
# 	resolution = 4 #[pix/m]
# 	angle_threshold = 1 # degrees

# 	radius_threshold_pix = radious_threshold * resolution #pix
# 	print "radius threshold", radius_threshold_pix
# 	is_line_merged = True
# 	while is_line_merged:
# 		merged_vine_rows = []
# 		vine_rows_to_delete = []
# 		number_of_end_points = len(vine_rows)

# 		is_line_merged = False
# 		# find end of rows points that are close to each other
# 		for i in range(0,number_of_end_points):
# 			for j in range(i,number_of_end_points):
# 				if i!=j:
# 					current_distance = distance.euclidean(vine_rows[i][:],vine_rows[j][:])
# 					#print "distance: ",current_distance
# 					if current_distance < radius_threshold_pix:
# 						#print "Near point found"
# 						angle_line_1 = angle_rows[i]
# 						angle_line_2 = angle_rows[j]

# 						#print "angles: ", angle_line_1, angle_line_2
# 						#print "difference", abs(angle_line_1-angle_line_2)
# 						# find it the line from the near point is aligned to the original
# 						if abs(angle_line_1-angle_line_2) < angle_threshold: 
							
# 							if i % 2 == 0:
# 								start_point_line_1 = vine_rows[i+1]
# 							else:
# 								start_point_line_1 = vine_rows[i-1]
# 							if j % 2 == 0:
# 								start_point_line_2 = vine_rows[j+1]
# 							else:
# 								start_point_line_2 = vine_rows[j-1]

# 							if (start_point_line_1[0]-start_point_line_2[0])==0:
# 								angle_merged_line = 90
# 							else:
# 								angle_merged_line = math.degrees(math.atan(float((start_point_line_1[1]-start_point_line_2[1]))/(start_point_line_1[0]-start_point_line_2[0])))

# 							# find if the resulting merged line angle is similar to the original
# 							if abs(angle_line_1-angle_merged_line) < angle_threshold:
# 								#print "Lines can be merged"
# 								merged_vine_rows.append(start_point_line_1)
# 								merged_vine_rows.append(start_point_line_2)
# 								if not (i in vine_rows_to_delete):
# 									vine_rows_to_delete.append(i)
# 									if i % 2 == 0:
# 										vine_rows_to_delete.append(i+1)
# 									else:
# 										vine_rows_to_delete.append(i-1)

# 								if not (j in vine_rows_to_delete):
# 									vine_rows_to_delete.append(j)
# 									if j % 2 == 0:
# 										vine_rows_to_delete.append(j+1)
# 									else:
# 										vine_rows_to_delete.append(j-1)

# 								is_line_merged = True

# 		#delete short lines that have been merged and append the merged ones
# 		for d in sorted(vine_rows_to_delete, reverse=True):
# 			del vine_rows[d]
# 		for f in range(0,len(merged_vine_rows)):
# 			vine_rows.append(merged_vine_rows[f])

# 		print "number of merged lines: ", len(merged_vine_rows)/2
# 		is_line_merged = False

if merge_row_lines:
	print "Trying to merge rows"
	# take the vine rows detected and try to recronstrucnt gaps in the row
	radious_threshold = 10 #meters
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
			org_line_angle = calculate_line_angle(vine_rows_full_line[org_line][:])#math.degrees(math.atan(float((vine_rows_full_line[org_line][1]-vine_rows_full_line[org_line][3]))/(vine_rows_full_line[org_line][0]-vine_rows_full_line[org_line][2])))
			for dest_line in range(org_line+1,number_of_lines):
				#check the angle difference between the two lines
				dest_line_angle = calculate_line_angle(vine_rows_full_line[dest_line][:])#math.degrees(math.atan(float((vine_rows_full_line[dest_line][1]-vine_rows_full_line[dest_line][3]))/(vine_rows_full_line[dest_line][0]-vine_rows_full_line[dest_line][2])))
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

				if is_line_merged == True:
					break
			if is_line_merged == True:
					break
		#delete short lines that have been merged and append the merged ones
		for d in sorted(lines_to_detele, reverse=True):
			del vine_rows_full_line[d]
			del angle_rows_full_line[d]
		vine_rows_full_line.append(line_to_merge)

# if compute_toponodes_locations:
# 	for row in v

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

	# for line in range(0,len(vine_rows),2):
	# 	axes.plot([vine_rows[line][0],vine_rows[line+1][0]],[vine_rows[line][1],vine_rows[line+1][1]], 'r', linewidth=3)

	for line in range(0,len(vine_rows_full_line)):
		axes.plot([vine_rows_full_line[line][0],vine_rows_full_line[line][2]],[vine_rows_full_line[line][1],vine_rows_full_line[line][3]], 'r', linewidth=3)	

	# axes[1].imshow(img_bin, cmap='gray')
	# axes[1].set_title('row lines')	
	# if merge_row_lines:
	# 	for line in range(0,len(merged_vine_rows),2):
	# 	 	axes.plot([merged_vine_rows[line][0],merged_vine_rows[line+1][0]],[merged_vine_rows[line][1],merged_vine_rows[line+1][1]], 'blue', linewidth=3)

	plt.show()