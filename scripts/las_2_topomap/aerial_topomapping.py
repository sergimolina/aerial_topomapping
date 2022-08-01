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
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from scipy.signal import argrelextrema
from skimage.filters import try_all_threshold
from skimage.color import rgb2hsv
from sklearn.decomposition import PCA
import math

from matplotlib import cm

from scipy import ndimage as ndi
from scipy.signal import argrelextrema
from scipy.spatial import distance
from scipy.stats import linregress

from shapely.geometry import LineString
from shapely.affinity import scale

import warnings
#from shapely.errors import ShapelyDeprecationWarning
from pyproj import Proj, transform
import rasterio
import datetime
import copy

def apply_binarisation(image):
	print("-- Applying binarisation --")
	print("-- Done --")
	return(image > 0)

def apply_morphological_operations(image):
	# Apply morphological operations
	print("-- Applying morphological operations --")
	# mask = circle(1,1,1)
	
	# image = binary_opening(image,mask)
	# image = binary_closing(image,mask)
	image = remove_small_objects(image,min_size=30)
	image = remove_small_holes(image)
	#image = segmentation.clear_border(image)
	# image = thin(image,1)
	print("-- Done --")
	return(image)

def row_classification(image,ratio_threshold):

	print("-- Applying row classification --")

	# Divide binary image in clusters by means of connectivity rules
	img_labels = measure.label(image,connectivity=2)
	num_of_clusters = np.max(img_labels)
	print("Total number of clusters:", num_of_clusters)

	img_only_rows = np.zeros([image.shape[0],image.shape[1]])
	number_of_clusters_after = 0
	ratios = np.array([])

	for cluster_num in range(1,num_of_clusters):

		# find all the x and y points beloging to the same cluster
		xx,yy = np.where(img_labels == cluster_num)
		X = np.array([xx,yy])
		X = X.T

		# compute PCA over the 2d features to obtain the eigen vectors describing the cluster
		pca = PCA(n_components=2).fit(X)
		ratio = pca.explained_variance_ratio_[0]/pca.explained_variance_ratio_[1]
		ratios = np.append(ratios,ratio)
		if ratio > ratio_threshold:
		 	img_only_rows[xx,yy] = 1 
		 	number_of_clusters_after = number_of_clusters_after + 1

	print("Number of rows detected:", number_of_clusters_after)
	print("-- Done --")

	return(img_only_rows)

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

def compute_row_lines(image):
	print("-- Calculating row lines --")

	# Divide binary image in clusters by means of connectivity rules
	img_labels = measure.label(image,connectivity=2)
	num_of_clusters = np.max(img_labels)
	vine_rows = []
	angle_rows = []
	vine_rows_full_line = []

	for current_cluster in range(1,num_of_clusters):#range(1,num_of_clusters):#clusters_to_test:#range(130,131):#num_of_clusters):

		# Compute lines for each cluster
		xx,yy = np.where(img_labels == current_cluster)
		img_cluster = np.zeros([image.shape[0],image.shape[1]])
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
		best_distances_ind = argrelextrema(intensities, np.greater,order = 5)
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


			x0 = int(x0)
			x1 = int(x1)
			y0 = int(y0)
			y1 = int(y1)

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
	
	print("Lines computed:",len(vine_rows_full_line))
	return(vine_rows_full_line)

def merge_row_lines(vine_rows_full_line,resolution,radious_threshold, angle_threshold):
	print("-- Trying to merge lines --")

	radius_threshold_pix = radious_threshold * resolution #pix
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
	print("Lines after merging:",len(vine_rows_full_line))
	print("-- Done --")
	return(vine_rows_full_line)


def compute_corridor_nodes(vine_rows_full_line, resolution, inter_row_distance, distance_between_nodes,distance_precorridor_nodes):
	#warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning) 
	print("-- Computing corridor nodes --")
	distance_between_rows_pix = inter_row_distance * resolution #pix
	distance_between_nodes_pix = distance_between_nodes * resolution # pix
	distance_precorridor_nodes_pix = distance_precorridor_nodes * resolution

	intra_corridor_topological_nodes = []
	outer_corridor_topological_nodes = []
	corridor_topological_nodes = []

	for row_number in range(0,len(vine_rows_full_line)):
		vine_rows_slope = calculate_line_slope(vine_rows_full_line[row_number])
		a = vine_rows_full_line[row_number][0:2]
		b = vine_rows_full_line[row_number][2:4]
		ab = LineString([a, b])
		left = ab.parallel_offset(distance_between_rows_pix/2, 'left')
		right = ab.parallel_offset(distance_between_rows_pix/2, 'right')

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

		corridor_topological_nodes.append([int(round(e_left.boundary[0].x)),  int(round(e_left.boundary[0].y)), int(round(left.boundary[0].x)),  int(round(left.boundary[0].y)), int(round(left.boundary[1].x)), int(round(left.boundary[1].y)), int(round(e_left.boundary[1].x)), int(round(e_left.boundary[1].y))])
		corridor_topological_nodes.append([int(round(e_right.boundary[0].x)), int(round(e_right.boundary[0].y)),int(round(right.boundary[0].x)), int(round(right.boundary[0].y)),int(round(right.boundary[1].x)),int(round(right.boundary[1].y)),int(round(e_right.boundary[1].x)), int(round(e_right.boundary[1].y))])

	print("-- Done --")
	return(corridor_topological_nodes)

def merge_corridor_nodes(corridor_topological_nodes, resolution, distance_threshold):
	print("-- Trying to merge corridor nodes --")
	distance_threshold_pix = distance_threshold * resolution #pix
	is_corridor_merged = True
	number_of_corridors = len(corridor_topological_nodes)
	print("Corridors before merging:",len(corridor_topological_nodes))

	while is_corridor_merged:
		number_of_corridors = len(corridor_topological_nodes)
		corridors_to_delete = []

		is_corridor_merged = False
		# find if the end of corridors nodes are close
		for c1 in range(0,number_of_corridors):
			c1_1 = corridor_topological_nodes[c1][2:4]
			c1_2 = corridor_topological_nodes[c1][4:6]
			for c2 in range(c1+1,number_of_corridors):
				c2_1 = corridor_topological_nodes[c2][2:4]
				c2_2 = corridor_topological_nodes[c2][4:6]
				if distance.euclidean(c1_1,c2_1)<distance_threshold_pix and distance.euclidean(c1_2,c2_2)<distance_threshold_pix:
					resulting_corridor = [(corridor_topological_nodes[c1][0]+corridor_topological_nodes[c2][0])/2,\
									      (corridor_topological_nodes[c1][1]+corridor_topological_nodes[c2][1])/2,\
									      (corridor_topological_nodes[c1][2]+corridor_topological_nodes[c2][2])/2,\
									      (corridor_topological_nodes[c1][3]+corridor_topological_nodes[c2][3])/2,\
									      (corridor_topological_nodes[c1][4]+corridor_topological_nodes[c2][4])/2,\
									      (corridor_topological_nodes[c1][5]+corridor_topological_nodes[c2][5])/2,\
									      (corridor_topological_nodes[c1][6]+corridor_topological_nodes[c2][6])/2,\
									      (corridor_topological_nodes[c1][7]+corridor_topological_nodes[c2][7])/2]
					if not (c1 in corridors_to_delete):
						corridors_to_delete.append(c1)
					if not (c2 in corridors_to_delete):
						corridors_to_delete.append(c2)
					is_corridor_merged = True

				elif distance.euclidean(c1_1,c2_2)<distance_threshold_pix and distance.euclidean(c1_2,c2_1)<distance_threshold_pix:
					resulting_corridor = [(corridor_topological_nodes[c1][0]+corridor_topological_nodes[c2][6])/2,\
									      (corridor_topological_nodes[c1][1]+corridor_topological_nodes[c2][7])/2,\
									      (corridor_topological_nodes[c1][2]+corridor_topological_nodes[c2][4])/2,\
									      (corridor_topological_nodes[c1][3]+corridor_topological_nodes[c2][5])/2,\
									      (corridor_topological_nodes[c1][4]+corridor_topological_nodes[c2][2])/2,\
									      (corridor_topological_nodes[c1][5]+corridor_topological_nodes[c2][3])/2,\
									      (corridor_topological_nodes[c1][6]+corridor_topological_nodes[c2][0])/2,\
									      (corridor_topological_nodes[c1][7]+corridor_topological_nodes[c2][1])/2]
					if not (c1 in corridors_to_delete):
						corridors_to_delete.append(c1)
					if not (c2 in corridors_to_delete):
						corridors_to_delete.append(c2)
					is_corridor_merged = True

				if is_corridor_merged:
					break
			if is_corridor_merged:
					break

		#delete corridors that have been merged and append the merged ones
		if is_corridor_merged:
			for c in sorted(corridors_to_delete, reverse=True):
				del corridor_topological_nodes[c]
			corridor_topological_nodes.append(resulting_corridor)


	print("Corridors after merging:",len(corridor_topological_nodes))
	print("-- Done --")
	return(corridor_topological_nodes)

def reproject_coordinates_to_lonlat(corridor_topological_nodes,crs,image_transform):
	print ("-- Transforming toponodes locations to longitude/latitude--")
	corridor_toponodes_data = {}
	corridor_toponodes_data['crs'] = 'epsg:4326'
	corridor_toponodes_data['corridors'] = []
	inProj = Proj(init=crs)
	outProj = Proj(init='epsg:4326') #world coordinates
	for c in corridor_topological_nodes:
		temp_corridor = []
		for p in range(0,8,2):
			temp_x,temp_y = rasterio.transform.xy(image_transform,rows=c[p+1],cols=c[p])
			x,y = transform(inProj,outProj,temp_x,temp_y)
			temp_corridor.append(x)
			temp_corridor.append(y)
		corridor_toponodes_data['corridors'].append(temp_corridor)
	print ("-- Done --")
	return corridor_toponodes_data

def reproject_coordinates_to_utm(corridor_topological_nodes,crs,image_transform):
	print ("-- Transforming toponodes locations to UTM --")
	corridor_toponodes_data = {}
	corridor_toponodes_data['crs'] = crs
	corridor_toponodes_data['corridors'] = []
	for c in corridor_topological_nodes:
		temp_corridor = []
		for p in range(0,8,2):
			x_utm,y_utm = rasterio.transform.xy(image_transform,rows=c[p+1],cols=c[p])
			temp_corridor.append(x_utm)
			temp_corridor.append(y_utm)
		corridor_toponodes_data['corridors'].append(temp_corridor)
	print ("-- Done --")
	return corridor_toponodes_data


def transform_toponodes_from_utm_to_map_coordinates(corridor_toponodes_utm,map_datum_longitude,map_datum_latitude):
	
	#get the datum in utm
	# inProj = Proj(init='epsg:4326')
	# outProj = Proj(init=corridor_toponodes_utm['crs']) #world coordinates
	# datum_x,datum_y = transform(inProj,outProj,datum_longitude,datum_latitude, always_xy=True)
	p=Proj(init=corridor_toponodes_utm['crs'])
	datum_x,datum_y = p(map_datum_longitude, map_datum_latitude)
	print(datum_x, datum_y)

	#calculte the toponoes in datum reference
	corridor_toponodes_map = {}
	corridor_toponodes_map['crs']= "custom_datum"
	corridor_toponodes_map['datum'] = {}
	corridor_toponodes_map['datum']['crs'] = corridor_toponodes_utm['crs']
	corridor_toponodes_map['datum']['longitude'] = datum_x
	corridor_toponodes_map['datum']['latitude'] = datum_y
	corridor_toponodes_map['corridors'] = []

	for c in corridor_toponodes_utm['corridors']:
		temp_corridor = []
		for p in range(0,8,2):
			#y_map = -c[p] + datum_x
			#x_map = -c[p+1] + datum_y
			x_map = c[p] - datum_x
			y_map = c[p+1] - datum_y			
			temp_corridor.append(x_map)
			temp_corridor.append(y_map)
		corridor_toponodes_map['corridors'].append(temp_corridor)

	return corridor_toponodes_map


def generate_topological_map(corridor_toponodes_map,tmap_name,template_toponode,template_topoedge):
	topomap = {}
	topomap["meta"] = {}
	topomap["meta"]["last_updated"] = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
	topomap["name"] = tmap_name
	topomap["metric_map"] = tmap_name
	topomap["pointset"] = tmap_name
	topomap["transformation"] = {}
	topomap["transformation"]["rotation"] = {}
	topomap["transformation"]["rotation"]["w"] = 1.0
	topomap["transformation"]["rotation"]["x"] = 0.0
	topomap["transformation"]["rotation"]["y"] = 0.0
	topomap["transformation"]["rotation"]["z"] = 0.0
	topomap["transformation"]["translation"] = {}
	topomap["transformation"]["translation"]["x"] = 0.0
	topomap["transformation"]["translation"]["y"] = 0.0
	topomap["transformation"]["translation"]["z"] = 0.0
	topomap["transformation"]["child"] = "topo_map"
	topomap["transformation"]["parent"] = "map"
	topomap["nodes"] = []

	for c in range(0,len(corridor_toponodes_map["corridors"])):
		num = 0
		for p in range(0,8,2):
			node = copy.deepcopy(template_toponode)
			node["meta"]["map"] = tmap_name 
			node["meta"]["pointset"] = tmap_name
			node["meta"]["node"] = "c"+str(c)+"_p"+str(num)
			node["node"]["name"] = "c"+str(c)+"_p"+str(num)
			node["node"]["pose"]["position"]["x"] = corridor_toponodes_map["corridors"][c][p] 
			node["node"]["pose"]["position"]["y"] = corridor_toponodes_map["corridors"][c][p+1]

			if num == 0:
				edge = copy.deepcopy(template_topoedge)
				edge["action"] = "row_traversal"
				edge["edge_id"] = "c"+str(c)+"_p"+str(num)+"-"+"c"+str(c)+"_p"+str(num+1)
				edge["node"] = "c"+str(c)+"_p"+str(num+1)
				node["node"]["edges"].append(edge)

			if num == 1 or num==2:
				edge = copy.deepcopy(template_topoedge)
				edge["action"] = "row_traversal"
				edge["edge_id"] = "c"+str(c)+"_p"+str(num)+"-"+"c"+str(c)+"_p"+str(num+1)
				edge["node"] = "c"+str(c)+"_p"+str(num+1)
				node["node"]["edges"].append(edge)

				edge = copy.deepcopy(template_topoedge)
				edge["action"] = "row_traversal"
				edge["edge_id"] = "c"+str(c)+"_p"+str(num)+"-"+"c"+str(c)+"_p"+str(num-1)
				edge["node"] = "c"+str(c)+"_p"+str(num-1)
				node["node"]["edges"].append(edge)

			if num == 3:
				edge = copy.deepcopy(template_topoedge)
				edge["action"] = "row_traversal"
				edge["edge_id"] = "c"+str(c)+"_p"+str(num)+"-"+"c"+str(c)+"_p"+str(num-1)
				edge["node"] = "c"+str(c)+"_p"+str(num-1)
				node["node"]["edges"].append(edge)

			topomap["nodes"].append(node)
			num = num+1
	return topomap