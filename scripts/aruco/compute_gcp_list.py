#!/usr/bin/env python
  
from __future__ import print_function # Python 2/3 compatibility
import cv2 # Import the OpenCV library
import numpy as np # Import Numpy library
import os
import sys
import getopt

desired_aruco_dictionary = "DICT_4X4_50"
 
# The different ArUco dictionaries built into the OpenCV library. 
ARUCO_DICT = {
  "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
  "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
  "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
  "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
  "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
  "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
  "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
  "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
  "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
  "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
  "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
  "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
  "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
  "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
  "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
  "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
  "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL
}
  
def main():
  """
  Main method of the program.
  """
  # Check that we have a valid ArUco marker
  if ARUCO_DICT.get(desired_aruco_dictionary, None) is None:
    print("[INFO] ArUCo tag of '{}' is not supported".format(
      args["type"]))
    sys.exit(0)
     
  # Load the ArUco dictionary
  print("[INFO] detecting '{}' markers...".format(
    desired_aruco_dictionary))
  this_aruco_dictionary = cv2.aruco.Dictionary_get(ARUCO_DICT[desired_aruco_dictionary])
  this_aruco_parameters = cv2.aruco.DetectorParameters_create()
  
  project_folder = "/home/sergi/workspace/aruco/test_map"

  # load the ID location file
  l = 0
  id_coordinates = {}
  #id_coordinates_file = open(id_coordinates_filename,"a")
  id_coordinates_file = open(os.path.join(project_folder,"id_coordinates.txt"),"r")
  for line in id_coordinates_file:
    if l==0: # the first line contain the name of the projection used for the geo coordinates.
      projection = line
    else:# the rest of the lines contain the gcp ids and their coordinates [id, longitude, latitude, altitude]
      current_line = line.split(' ')
      _id = int(current_line[0])
      _id_longitude = float(current_line[1])
      _id_latitude = float(current_line[2])
      _id_altitude =float(current_line[3])

      id_coordinates[_id] = [_id_longitude,_id_latitude,_id_altitude]
    l=l+1
  id_coordinates_file.close()

  # create the ground control points list file
  gcp_list_file = open(os.path.join(project_folder,"images/gcp_list.txt"), "w")
  # write in the first line the name of the projection used for the geo coordinates.
  gcp_list_file.write(projection)

  # read and process all the images from the folder
  images_folder_path = os.path.join(project_folder,"images")
  for img_filename in os.listdir(images_folder_path):
    img = cv2.imread(os.path.join(images_folder_path,img_filename))
    if img is not None: #we can read the image

      # Detect ArUco markers in the image
      (corners, ids, rejected) = cv2.aruco.detectMarkers(img, this_aruco_dictionary, parameters=this_aruco_parameters)

      # Check that at least one ArUco marker was detected
      if len(corners) > 0:

        # Flatten the ArUco IDs list
        ids = ids.flatten()
         
        # Loop over the detected ArUco corners
        for (marker_corner, marker_id) in zip(corners, ids):
         
          # Extract the marker corners
          corners = marker_corner.reshape((4, 2))
          (top_left, top_right, bottom_right, bottom_left) = corners
           
          # Calculate and  the center of the ArUco marker in the image
          center_x = int((top_left[0] + bottom_right[0]) / 2.0)
          center_y = int((top_left[1] + bottom_right[1]) / 2.0)

          #save the detection in the detections ground control point list
          gcp_list_file.write(str(id_coordinates[int(marker_id)][0])) # longitude
          gcp_list_file.write(" ")
          gcp_list_file.write(str(id_coordinates[int(marker_id)][1])) # latitude
          gcp_list_file.write(" ")
          gcp_list_file.write(str(id_coordinates[int(marker_id)][2])) # altitude
          gcp_list_file.write(" ")
          gcp_list_file.write(str(center_x)) # pixel coordinates
          gcp_list_file.write(" ")
          gcp_list_file.write(str(center_y)) # pixel coordinates
          gcp_list_file.write(" ")
          gcp_list_file.write(img_filename) # file name of the picture
          gcp_list_file.write("\n")

  gcp_list_file.close()
   
if __name__ == '__main__':
  main()