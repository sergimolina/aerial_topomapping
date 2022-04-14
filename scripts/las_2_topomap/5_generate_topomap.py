import rasterio
from rasterio.plot import show
import aerial_topomapping as at
import matplotlib.pyplot as plt
import os
import numpy as np
import json
import yaml
from pyproj import Proj, transform
import aerial_topomapping as at


##########################
toponodes_filename = '../../data/riseholme_correction/riseholme_correction_toponodes_map.json'
#datum_longitude = -0.524509505881
#datum_latitude = 53.268642038
#datum_longitude = 22.9243
#datum_latitude = 40.45025
tmap_name = "riseholme_uav_navsat"

#read the toponodes file
with open(toponodes_filename, 'r') as f:
	corridor_toponodes_map = json.load(f)

#load node and edges templates
with open("template_toponode.json", 'r') as f:
	template_toponode = json.load(f)

with open("template_topoedge.json", 'r') as f:
	template_topoedge = json.load(f)

# transform the geo located nodes to the ros map coordinates  
#corridor_toponodes_map = at.transform_toponodes_from_utm_to_map_coordinates(corridor_toponodes_utm,datum_longitude,datum_latitude)
#corridor_toponodes_map = at.transform_toponodes_from_lonlat_to_map_coordinates_with_ros_navsat(corridor_toponodes_lonlat)


topomap = at.generate_topological_map(corridor_toponodes_map,tmap_name,template_toponode,template_topoedge)
#topomap = at.generate_topological_map_in_utm(corridor_toponodes_utm,tmap_name,template_toponode)
with open(tmap_name+".tmap2",'w') as f:
	yaml.dump(topomap, f)
