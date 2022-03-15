#!/usr/bin/env python

import rospy
import yaml
from sensor_msgs.msg import CameraInfo
import math

class camera_info_publisher(object):

	def __init__(self):
		#parameters
		self.camera_info_file = rospy.get_param('~camera_info_file',"info_file")

		#publishers
		self.camera_info_pub = rospy.Publisher("/camera_info", CameraInfo, queue_size=10)

		# read parameters and publish them
		with open(self.camera_info_file, "r") as file_handle:
			calib_data = yaml.load(file_handle)

		# Parse
		self.camera_info = CameraInfo()
		# store info without header
		self.camera_info.width = calib_data["image_width"]
		self.camera_info.height = calib_data["image_height"]
		self.camera_info.distortion_model = calib_data["distortion_model"]
		cx = self.camera_info.width / 2.0
		cy = self.camera_info.height / 2.0
		fx = self.camera_info.width / (2.0 * math.tan(calib_data["fov"] * math.pi / 360.0))
		fy = fx
		self.camera_info.K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
		self.camera_info.D = [0, 0, 0, 0, 0]
		self.camera_info.R = [1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0]
		self.camera_info.P = [fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1.0, 0]

		self.run()


	def run(self):
		r = rospy.Rate(1)
		while not rospy.is_shutdown():
			self.camera_info.header.stamp = rospy.get_rostime()
			self.camera_info.header.frame_id = "camera"
			self.camera_info_pub.publish(self.camera_info)
			r.sleep()

if __name__ == '__main__':
	rospy.init_node('camera_info_publisher_node', anonymous=True)
	camera_info_publisher = camera_info_publisher()