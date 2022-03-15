#!/usr/bin/env python3

import laspy
from PIL import Image
import numpy as np

def get_real_x_points(las_file):
    x_dimension = las_file.X
    scale = las_file.header.scales[0]
    offset = las_file.header.offsets[0]
    return (x_dimension * scale) + offset

def get_real_y_points(las_file):
    y_dimension = las_file.Y
    scale = las_file.header.scales[1]
    offset = las_file.header.offsets[1]
    return (y_dimension * scale) + offset

def get_real_z_points(las_file):
    z_dimension = las_file.Z
    scale = las_file.header.scales[2]
    offset = las_file.header.offsets[2]
    return (z_dimension * scale) + offset

las_file = laspy.read('../data/pointclouds/ktima_Gerovasileioy_2020-07-21_Field_90_group1_densified_point_cloud_part_7.las')
resolution = 0.1

points_x = get_real_x_points(las_file)
points_y = get_real_y_points(las_file)
points_z = get_real_z_points(las_file)

min_x=min(points_x)
max_x=max(points_x)
min_y=min(points_y)
max_y=max(points_y)
min_z=min(points_z)
max_z=max(points_z)

points_x = points_x - min_x
points_y = points_y - min_y
points_z = points_z - min_z

points_z = (points_z/(max_z-min_z))*255

min_x=min(points_x)
max_x=max(points_x)
min_y=min(points_y)
max_y=max(points_y)
min_z=min(points_z)
max_z=max(points_z)

# CONVERT TO PIXEL POSITION VALUES - Based on resolution
x_img = (points_x/resolution).astype(np.int32) # x axis is -y in LIDAR
y_img = (points_y/resolution).astype(np.int32)  # y axis is -x in LIDAR

# FILL PIXEL VALUES IN IMAGE ARRAY
x_max = int(max(points_x)/resolution)
y_max = int(max(points_y)/resolution)
im = np.zeros([y_max, x_max], dtype=np.uint8)
im[-y_img, -x_img] = points_z # -y because images start from top left

im = Image.fromarray(im)

im.show()
# def scaled_x_dimension(las_file):
#     x_dimension = las_file.X
#     scale = las_file.header.scales[0]
#     offset = las_file.header.offsets[0]
#     return (x_dimension * scale) + offset

# scaled_x = scaled_x_dimension(las)

# print(scaled_x)