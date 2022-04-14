import rospy
from geographic_msgs.msg import GeoPoint
from robot_localization.srv import FromLL
import json
import yaml

def transform_toponodes_from_lonlat_to_map_coordinates_with_ros_navsat(corridor_toponodes_lonlat):
	
	print ("-- Transforming toponodes locations from Lon/Lat to Map coordinates with ROS Navsat Node --")
	print ("Waiting for the 'fromLL' rosservice to be available...")
	rospy.wait_for_service("fromLL")
	print("Rosservice ready. Computing the transforms for all the nodes...")
	
	corridor_toponodes_map = {}
	#corridor_toponodes_map['crs']= "custom_datum"
	#corridor_toponodes_map['datum'] = {}
	#corridor_toponodes_map['datum']['crs'] = corridor_toponodes_lonlat['crs']
	#corridor_toponodes_map['datum']['longitude'] = datum_x
	#corridor_toponodes_map['datum']['latitude'] = datum_y
	corridor_toponodes_map['corridors'] = []


	try:
		fromLL_service = rospy.ServiceProxy('fromLL', FromLL)
		for c in corridor_toponodes_lonlat['corridors']:
			temp_corridor = []
			for p in range(0,8,2):
				geopoint_msg = GeoPoint()
				geopoint_msg.longitude = c[p]
				geopoint_msg.latitude = c[p+1]
				resp = fromLL_service(geopoint_msg)
				x_map = resp.map_point.x
				y_map = resp.map_point.y
				temp_corridor.append(x_map)
				temp_corridor.append(y_map)
			corridor_toponodes_map['corridors'].append(temp_corridor)
	except rospy.ServiceException as e:
		print("Service call failed: %s"%e)		
	print("-- Done --")
	return corridor_toponodes_map


##########################
input_toponodes_lonlat_filename = '../../data/riseholme_correction/riseholme_correction_toponodes_lonlat.json'
ouput_toponodes_map_filename = '../../data/riseholme_correction/riseholme_correction_toponodes_map.json'

#read the toponodes file
with open(input_toponodes_lonlat_filename, 'r') as f:
	corridor_toponodes_lonlat = json.load(f)

# transform the geo located nodes to the ros map coordinates 
corridor_toponodes_map = transform_toponodes_from_lonlat_to_map_coordinates_with_ros_navsat(corridor_toponodes_lonlat)

with open(ouput_toponodes_map_filename,'w') as f:
	json.dump(corridor_toponodes_map, f,indent=4)
