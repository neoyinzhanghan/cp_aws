# rainbow_heatmap_path_npy = '/media/ssd2/neo/cp_aws_playground/23.CFNA.113 A1 H&E _171848_rainbow_heatmap_mask.npy'
wsi_path = '/media/ssd2/neo/cp_aws_playground/23.CFNA.113 A1 H&E _171848.svs'
coolwarm_heatmap_path_npy = '/media/ssd2/neo/cp_aws_playground/23.CFNA.113 A1 H&E_coolwarm_heatmap_mask.npy'
coolwarm_heatmap_path_png = '/media/ssd2/neo/cp_aws_playground/23.CFNA.113 A1 H&E_coolwarm_heatmap_mask.png'
rainbow_heatmap_path_npy = '/media/ssd2/neo/cp_aws_playground/23.CFNA.113 A1 H&E_rainbow_heatmap_mask.npy'
rainbow_heatmap_path_png = '/media/ssd2/neo/cp_aws_playground/23.CFNA.113 A1 H&E_rainbow_heatmap_mask.png'

import os
import openslide
import numpy as np
from PIL import Image

############################################################################
### WSI INFO
############################################################################

# Load the WSI
wsi = openslide.OpenSlide(wsi_path)

# Get number of levels
num_levels = wsi.level_count

print(f"Number of levels: {num_levels}")

# Print dimensions and MPP at each level
for level in range(num_levels):
    dimensions = wsi.level_dimensions[level]
    downsample = wsi.level_downsamples[level]
    print(f"Level {level}:")
    print(f"  Dimensions: {dimensions}")
    print(f"  Downsample factor: {downsample}")
    if "openslide.mpp-x" in wsi.properties and "openslide.mpp-y" in wsi.properties:
        mpp_x = float(wsi.properties["openslide.mpp-x"]) * downsample
        mpp_y = float(wsi.properties["openslide.mpp-y"]) * downsample
        print(f"  MPP (X, Y): ({mpp_x:.3f}, {mpp_y:.3f}) microns/pixel")
    else:
        print("  MPP information not available.")

# Close the WSI to free resources
wsi.close()

############################################################################
### NPY INFO
############################################################################

# open the coolwarm heatmap as a numpy array
coolwarm_heatmap = np.load(coolwarm_heatmap_path_npy)

print(f"Coolwarm heatmap shape: {coolwarm_heatmap.shape}")

# open the rainbow heatmap as a numpy array
rainbow_heatmap = np.load(rainbow_heatmap_path_npy)

print(f"Rainbow heatmap shape: {rainbow_heatmap.shape}")

############################################################################
# PNG INFO
############################################################################

# check if the PNG files exist
coolwarm_heatmap_png_exists = os.path.exists(coolwarm_heatmap_path_png)

print(f"Coolwarm heatmap PNG exists: {coolwarm_heatmap_png_exists}")

rainbow_heatmap_png_exists = os.path.exists(rainbow_heatmap_path_png)

print(f"Rainbow heatmap PNG exists: {rainbow_heatmap_png_exists}")

# now get the dimension of the PNG files

# if coolwarm_heatmap_png_exists:
#     coolwarm_heatmap_png = Image.open(coolwarm_heatmap_path_png)
#     coolwarm_heatmap_png_dimensions = coolwarm_heatmap_png.size
#     print(f"Coolwarm heatmap PNG dimensions: {coolwarm_heatmap_png_dimensions}")
#     coolwarm_heatmap_png.close()   

# if rainbow_heatmap_png_exists:
#     rainbow_heatmap_png = Image.open(rainbow_heatmap_path_png)
#     rainbow_heatmap_png_dimensions = rainbow_heatmap_png.size
#     print(f"Rainbow heatmap PNG dimensions: {rainbow_heatmap_png_dimensions}")


############################################################################
# FILE SIZE ANALYSIS -- in MB
############################################################################

print(f"WSI file size: {os.path.getsize(wsi_path) / 1024 / 1024:.2f} MB")
print(f"Coolwarm heatmap file size: {os.path.getsize(coolwarm_heatmap_path_npy) / 1024 / 1024:.2f} MB")
print(f"Rainbow heatmap file size: {os.path.getsize(rainbow_heatmap_path_npy) / 1024 / 1024:.2f} MB")
if coolwarm_heatmap_png_exists:
    print(f"Coolwarm heatmap PNG file size: {os.path.getsize(coolwarm_heatmap_path_png) / 1024 / 1024:.2f} MB")
if rainbow_heatmap_png_exists:
    print(f"Rainbow heatmap PNG file size: {os.path.getsize(rainbow_heatmap_path_png) / 1024 / 1024:.2f} MB")