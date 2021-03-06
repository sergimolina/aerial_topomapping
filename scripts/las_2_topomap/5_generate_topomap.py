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
tmap_name = "riseholme_uav_navsat"

#read the toponodes file
with open(toponodes_filename, 'r') as f:
	corridor_toponodes_map = json.load(f)

#load node and edges templates
with open("template_toponode.yaml", 'r') as f:
	template_toponode = yaml.safe_load(f)

with open("template_topoedge.yaml", 'r') as f:
	template_topoedge = yaml.safe_load(f)

topomap = at.generate_topological_map(corridor_toponodes_map,tmap_name,template_toponode,template_topoedge)

with open('../../data/riseholme_correction/'+tmap_name+".tmap2",'w') as f:
	yaml.dump(topomap, f)
