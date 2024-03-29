import rasterio
from rasterio.plot import show
import aerial_topomapping as at
import matplotlib.pyplot as plt
import os
import numpy as np
import json

#image_filename = 'KG_big1.tif'
#image_filename = '../../data/KG/KG.tif'
image_filename = '../../data/riseholme_correction/riseholme_correction.tif'
image_resolution = 10 #[pix/m]
cluster_ratio_threshold = 30
radious_threshold = 5 #[m]
angle_threshold = 1 #[degrees]
inter_row_distance = 2 #[m]
distance_between_nodes = 5 #[m]
distance_precorridor_nodes = 1 #[m]
merge_corridor_distance_threshold = 1.5 #[m]
refresh = True


image = rasterio.open(image_filename)
band1 = image.read(1)
print(image.transform)
# obtain the clusters classified as canopy rows
if os.path.isfile(image_filename[:-4]+'_rows_clusters_img.npy') and refresh==False:
	print ("-- Loading row clusters image from file --")
	band1_mod = np.load(image_filename[:-4]+'_rows_clusters_img.npy')
else:
	band1_mod = at.apply_binarisation(band1)
	band1_mod = at.apply_morphological_operations(band1_mod)
	band1_mod = at.row_classification(band1_mod,cluster_ratio_threshold)
	np.save(image_filename[:-4]+'_rows_clusters_img',band1_mod)

# Compute the canopy row lines
if os.path.isfile(image_filename[:-4]+'_row_lines.npy') and refresh==False:
	print ("-- Loading row lines from file --")
	row_lines = np.load(image_filename[:-4]+'_row_lines.npy')
else:
	row_lines = at.compute_row_lines(band1_mod)
	row_lines = at.merge_row_lines(row_lines,image_resolution, radious_threshold , angle_threshold)
	np.save(image_filename[:-4]+'_row_lines',row_lines)

# Compute the corridor topological nodes
corridor_toponodes_pix = at.compute_corridor_nodes(row_lines, image_resolution, inter_row_distance, distance_between_nodes,distance_precorridor_nodes)
corridor_toponodes_pix = at.merge_corridor_nodes(corridor_toponodes_pix, image_resolution, merge_corridor_distance_threshold)

corridor_toponodes_utm = at.reproject_coordinates_to_utm(corridor_toponodes_pix,str(image.crs),image.transform)
with open(image_filename[:-4]+'_toponodes_utm.json','w') as f:
	json.dump(corridor_toponodes_utm,f,indent=4)

corridor_toponodes_lonlat = at.reproject_coordinates_to_lonlat(corridor_toponodes_pix,str(image.crs),image.transform)
with open(image_filename[:-4]+'_toponodes_lonlat.json','w') as f:
	json.dump(corridor_toponodes_lonlat,f,indent=4)

#plotting
fig, axes = plt.subplots(nrows=1, ncols=1)
axes.imshow(band1,cmap="gray")
for c in corridor_toponodes_pix:
	axes.plot(c[0],c[1],'ro')
	axes.plot(c[2],c[3],'bo')
	axes.plot(c[4],c[5],'bo')
	axes.plot(c[6],c[7],'ro')
for r in row_lines:
	axes.plot(r[0],r[1],'go')
	axes.plot(r[2],r[3],'go')
plt.show()