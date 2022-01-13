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

def compute_windows_histogram(greyscale_image,window_size,image_file_name):
	print "Computing the histogram for each window..."

	if os.path.isfile(image_file_name[:-4]+'_windows_histogram.npy'):
		windows_histogram = np.load(image_file_name[:-4]+'_windows_histogram.npy')
		windows_color_mean = np.load(image_file_name[:-4]+'_windows_color_mean.npy')
	else:
		# old way
		#pixels_histograms = rank.windowed_histogram(greyscale_image,square(window_size),n_bins=256)
		#pixels_histograms = pixels_histograms*window_num_pixels
		#pixels_histograms = pixels_histograms.astype(int)
		#print pixels_histograms.shape

		rows = greyscale_image.shape[0]
		cols = greyscale_image.shape[1]

		windows_histogram = np.zeros([rows,cols,256])
		windows_color_mean = np.zeros([rows,cols])
		for row in range(0,rows):
			for col in range(0,cols):
				startx = int(row-0.5*window_size)
				starty = int(col-0.5*window_size)
				endx = int(row+0.5*window_size)
				endy = int(col+0.5*window_size)
				rr,cc = rectangle((startx,starty), end=(endx,endy), shape=greyscale_image.shape)
				pixel_values = greyscale_image[rr,cc]
				pixel_values_1d = np.reshape(pixel_values,(pixel_values.shape[0]*pixel_values.shape[1]))
				windows_color_mean[row,col] = np.mean(pixel_values)
				for c in range(0,256):
					windows_histogram[row,col,c] = np.count_nonzero(pixel_values_1d == c)
			print "computing windows histogram: " + str(float(100*row/rows))+"%"
		
		np.save(image_file_name[:-4]+'_windows_histogram',windows_histogram)
		np.save(image_file_name[:-4]+'_windows_color_mean',windows_color_mean)

	print "Done"
	return windows_histogram, windows_color_mean

def compute_windows_moment(windows_histogram, windows_color_mean, window_num_pixels):
	print "Computing the moment for each window..."

	if os.path.isfile(image_file_name[:-4]+'_windows_moment.npy'):
		windows_moment = np.load(image_file_name[:-4]+'_windows_moment.npy')
	else:
		rows = windows_histogram.shape[0]
		cols = windows_histogram.shape[1]
		window_num_pixels = int(window_size*window_size)
		windows_moment = np.zeros([rows,cols])
		for row in range(0,rows):
			for col in range(0,cols):
				for color in range(0,256):
					windows_moment[row,col] = windows_moment[row,col] + (windows_histogram[row,col,color]*(color-windows_color_mean[row,col])*(color-windows_color_mean[row,col]))
				windows_moment[row,col] = windows_moment[row,col]/window_num_pixels
			print "computing windows moment: " + str(float(100*row/rows))+"%"
		
		np.save(image_file_name[:-4]+'_windows_moment',windows_moment)
	print "Done"
	return windows_moment

def compute_pixels_count(greyscale_image, windows_moment, window_size):
	print "Computing the pixel counts for each window..."

	if os.path.isfile(image_file_name[:-4]+'_pixels_count.npy'):
		pixels_count = np.load(image_file_name[:-4]+'_pixels_count.npy')
		pixels_times_seen = np.load(image_file_name[:-4]+'_pixels_times_seen.npy')
	else:
		rows = greyscale_image.shape[0]
		cols = greyscale_image.shape[1]
		pixels_count = np.zeros([rows,cols])
		pixels_times_seen = np.zeros([rows,cols])
		for row in range(0,rows):
			for col in range(0,cols):
				if windows_moment[row,col] > moment_threshold: #only apply counts with window moent over a threshold
					# pixels_count[row,col] = 255
					startx = int(row-0.5*window_size)
					starty = int(col-0.5*window_size)
					endx = int(row+0.5*window_size)
					endy = int(col+0.5*window_size)
					rr,cc = rectangle((startx,starty), end=(endx,endy), shape=greyscale_image.shape)
					pixels_times_seen[rr,cc] = pixels_times_seen[rr,cc] + 1
					pixels_value = greyscale_image[rr,cc]
					ii,jj =  np.where(pixels_value > windows_color_mean[row,col])
					pixels_count[rr[ii,jj],cc[ii,jj]] = pixels_count[rr[ii,jj],cc[ii,jj]] + 1

			print "Computing pixels count: " + str(float(100*row/rows))+"%"
		np.save(image_file_name[:-4]+'_pixels_count',pixels_count)
		np.save(image_file_name[:-4]+'_pixels_times_seen',pixels_times_seen)
		
	print "Done"
	return pixels_count, pixels_times_seen

def compute_pixels_count_filter(pixels_count,pixels_times_seen, pixel_count_threshold):
	print "Filtering counts image with threshold"
	img_bin = np.zeros([pixels_count.shape[0],pixels_count.shape[1]])
	for row in range(0,pixels_count.shape[0]):
		for col in range(0,pixels_count.shape[1]):
			if pixels_count[row,col] > (pixel_count_threshold*pixels_times_seen[row,col]):
				img_bin[row,col] = 1
	img_bin = img_bin.astype(int)
	print "Done"
	return img_bin

def rgb_to_ExG_index(rgb_image):
	print "Getting ExG"
	#ExG = 2*g-r-b
	#r=R/(R+G+B),g=G/(R+G+B),b=B/(R+G+B)
	if os.path.isfile(image_file_name[:-4]+'_ExG.npy'):
		ExG_image = np.load(image_file_name[:-4]+'_ExG.npy')
	else:
		rows = rgb_image.shape[0]
		cols = rgb_image.shape[1]

		ExG_image = np.zeros([rows,cols])
		for row in range(0,rows):
			for col in range(0,cols):
				if sum(rgb_image[row,col,:]) > 0:
					r = np.true_divide(rgb_image[row,col,0],sum(rgb_image[row,col,:]))
					g = np.true_divide(rgb_image[row,col,1],sum(rgb_image[row,col,:]))
					b = np.true_divide(rgb_image[row,col,2],sum(rgb_image[row,col,:]))
					ExG_image[row,col] = 2*g-r-b

		#normalize between 0-255
		ExG_image = ((ExG_image - ExG_image.min()) * (1/(ExG_image.max() - ExG_image.min()) * 255)).astype('uint8')
		np.save(image_file_name[:-4]+'_ExG',ExG_image)

	return ExG_image

def rgb_to_GLI_index(rgb_image):
	print "Getting GLI"
	if os.path.isfile(image_file_name[:-4]+'_GLI.npy'):
		GLI_image = np.load(image_file_name[:-4]+'_GLI.npy')
	else:
		rows = rgb_image.shape[0]
		cols = rgb_image.shape[1]

		GLI_image = np.zeros([rows,cols])
		for row in range(0,rows):
			for col in range(0,cols):
				if sum(rgb_image[row,col,:]) > 0:
					r = np.true_divide(rgb_image[row,col,0],sum(rgb_image[row,col,:]))
					g = np.true_divide(rgb_image[row,col,1],sum(rgb_image[row,col,:]))
					b = np.true_divide(rgb_image[row,col,2],sum(rgb_image[row,col,:]))
					GLI_image[row,col] = (2*g-r-b)/(2*g+r+b)

		#normalize between 0-255
		GLI_image = ((GLI_image - GLI_image.min()) * (1/(GLI_image.max() - GLI_image.min()) * 255)).astype('uint8')
		np.save(image_file_name[:-4]+'_GLI',GLI_image)

	return GLI_image

def rgb_to_NDI_index(rgb_image):
	print "Getting NDI"
	if os.path.isfile(image_file_name[:-4]+'_NDI.npy'):
		NDI_image = np.load(image_file_name[:-4]+'_NDI.npy')
	else:
		rows = rgb_image.shape[0]
		cols = rgb_image.shape[1]

		NDI_image = np.zeros([rows,cols])
		for row in range(0,rows):
			for col in range(0,cols):
				if sum(rgb_image[row,col,:]) > 0:
					r = np.true_divide(rgb_image[row,col,0],sum(rgb_image[row,col,:]))
					g = np.true_divide(rgb_image[row,col,1],sum(rgb_image[row,col,:]))
					b = np.true_divide(rgb_image[row,col,2],sum(rgb_image[row,col,:]))
					NDI_image[row,col] = (g-r)/(g+r)

		#normalize between 0-255
		NDI_image = ((NDI_image - NDI_image.min()) * (1/(NDI_image.max() - NDI_image.min()) * 255)).astype('uint8')
		np.save(image_file_name[:-4]+'_NDI',NDI_image)

	return NDI_image

def rgb_to_VARI_index(rgb_image):
	print "Getting VARI"
	if os.path.isfile(image_file_name[:-4]+'_VARI.npy'):
		VARI_image = np.load(image_file_name[:-4]+'_VARI.npy')
	else:
		rows = rgb_image.shape[0]
		cols = rgb_image.shape[1]

		VARI_image = np.zeros([rows,cols])
		for row in range(0,rows):
			for col in range(0,cols):
				if sum(rgb_image[row,col,:]) > 0:
					r = np.true_divide(rgb_image[row,col,0],sum(rgb_image[row,col,:]))
					g = np.true_divide(rgb_image[row,col,1],sum(rgb_image[row,col,:]))
					b = np.true_divide(rgb_image[row,col,2],sum(rgb_image[row,col,:]))
					VARI_image[row,col] = (g-r)/(g+r-b)

		#normalize between 0-255
		VARI_image = ((VARI_image - VARI_image.min()) * (1/(VARI_image.max() - VARI_image.min()) * 255)).astype('uint8')
		np.save(image_file_name[:-4]+'_VARI',VARI_image)

	return VARI_image

#Parameters
image_file_name = './data/greece_zoom20_23cmpix.jpg'
img_resolution = 0.23 # m/pix
row_separation = 2.5 #meters
window_times_row = 5 #times
moment_threshold = 100
pixel_count_threshold = 0.85 # thershold to segment between canopy vs rest
vegetation_index = 0 # 0:ExG, 1:GLI, 2:2G_RBi, 3:ExGR, 4:NDI , 5:G%, 6:VARI, 7:HSV

compute_segmentation = False
show_plots = True

# Read the image
img_org = io.imread(image_file_name)
print('Original Dimensions : ',img_org.shape) 


# Transform RGB to a vegetation index
if vegetation_index == 0:
	img_grey = rgb_to_ExG_index(img_org)
if vegetation_index == 1:
	img_grey = rgb_to_GLI_index(img_org)
if vegetation_index == 2:
	img_grey = rgb_to_2G_RBi_index(img_org)
if vegetation_index == 3:
	img_grey = rgb_to_ExGR_index(img_org)
if vegetation_index == 4:
	img_grey = rgb_to_NDI_index(img_org)
if vegetation_index == 5:
	img_grey = rgb_to_Gper_index(img_org)
if vegetation_index == 6:
	img_grey = rgb_to_VARI_index(img_org)

print('Grey Dimensions : ',img_grey.shape)

fig, ax = try_all_threshold(img_grey, figsize=(10, 8), verbose=False)
plt.show()




if compute_segmentation:
	# Calculate the window size based on the row separtion and the times specified
	window_size = int(np.ceil( np.ceil(window_times_row*row_separation/img_resolution) / 2.) * 2) #round up to the nearest even integer

	# Compute the histograms for each pixel with the window size defined
	windows_histogram, windows_color_mean = compute_windows_histogram(img_grey,window_size, image_file_name)

	# Compute the windows moment
	windows_moment = compute_windows_moment(windows_histogram, windows_color_mean, window_size)

	# Compute the counts on each pixel to segment the vine vs background
	pixels_count, pixels_times_seen = compute_pixels_count(img_grey, windows_moment, window_size)

	# Apply filter to get a binary image
	img_bin_org = compute_pixels_count_filter(pixels_count,pixels_times_seen, pixel_count_threshold)


if show_plots:
	# #draw the shape in the image to check the size
	# px = 100
	# py = 100
	# rr,cc = line(px,py,px+window_size,py)
	# img_org[rr,cc] = (255,0,0)
	# rr,cc = line(px+window_size,py,px+window_size,py+window_size)
	# img_org[rr,cc] = (255,0,0)
	# rr,cc = line(px+window_size,py+window_size,px,py+window_size)
	# img_org[rr,cc] = (255,0,0)
	# rr,cc = line(px,py+window_size,px,py)
	# img_org[rr,cc] = (255,0,0)

	fig, axes = plt.subplots(nrows=1, ncols=2)
	axes[0].imshow(img_org)
	axes[0].set_title('original')	

	axes[1].imshow(img_grey, cmap='gray')
	axes[1].set_title('grey')

	# axes[0,2].set_title('binary')
	# axes[0,2].imshow(img_bin_org, cmap='gray')

	# axes[1, 0].set_title('counts')
	# axes[1, 0].imshow(pixels_count, cmap='gray')

	# axes[0,2].set_title('bin open close remove')
	# axes[0,2].imshow(img_bin_mod)

	# axes[1,0].set_title('lables')
	# axes[1,0].imshow(img_labels, cmap='nipy_spectral')

	# axes[1,1].set_title('cluster')
	# axes[1,1].imshow(img_cluster)
	# axes[1,1].plot(line_y, line_x, '-k', label='Line model from all data')
	# axes[1,1].plot(line_y_robust, line_x, '-b', label='Robust line model')
	

	# axes[1,1].imshow(img_cluster)
	# axes[1,1].plot((lines[l][0][0], lines[l][1][0]), (lines[l][0][1], lines[l][1][1]))
	# axes[1,1].set_xlim((0, img_cluster.shape[1]))
	# axes[1,1].set_ylim((img_cluster.shape[0], 0))
	# axes[1,1].set_title('Probabilistic Hough')

	plt.tight_layout()
	plt.show()