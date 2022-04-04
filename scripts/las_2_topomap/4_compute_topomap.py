import rasterio
from rasterio.plot import show
import aerial_topomapping as at
import matplotlib.pyplot as plt
import os
import numpy as np
import json
import yaml
from pyproj import Proj, transform
import datetime
import copy

def transform_toponodes_from_utm_to_map_coordinates(corridor_toponodes_utm,datum_longitude,datum_latitude):
	
	#get the datum in utm
	inProj = Proj(init='epsg:4326')
	outProj = Proj(init=corridor_toponodes_utm['crs']) #world coordinates
	datum_x,datum_y = transform(inProj,outProj,datum_longitude,datum_latitude)

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
			y = -c[p] + datum_x
			x = -c[p+1] + datum_y
			temp_corridor.append(x)
			temp_corridor.append(y)
		corridor_toponodes_map['corridors'].append(temp_corridor)

	return corridor_toponodes_map

def generate_topological_map(corridor_toponodes_map,tmap_name,template_toponode):
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

			topomap["nodes"].append(node)
			num = num+1
	return topomap

##########################
toponodes_filename = '../../data/riseholme/riseholme_toponodes_utm.json'
datum_longitude = -0.524509505881
datum_latitude = 53.268642038
tmap_name = "riseholme_test"

#read the toponodes file
with open(toponodes_filename, 'r') as f:
	corridor_toponodes_utm = json.load(f)

#load node and edges templates
with open("template_toponode.json", 'r') as f:
	template_toponode = json.load(f)

with open("template_topoedge.json", 'r') as f:
	template_topoedge = json.load(f)

corridor_toponodes_map = transform_toponodes_from_utm_to_map_coordinates(corridor_toponodes_utm,datum_longitude,datum_latitude)
topomap = generate_topological_map(corridor_toponodes_map,tmap_name,template_toponode)
with open(tmap_name+".tmap2",'w') as f:
	yaml.dump(topomap, f)
