import time
import openslide
from PIL import Image

wsi_path = '/media/ssd2/neo/cp_aws_playground/23.CFNA.113 A1 H&E _171848.svs'

print("Getting the level 0 image of the WSI...")
start_time = time.time()
# get the entire level 0 image of the wsi
wsi = openslide.OpenSlide(wsi_path)
level = 0

# get the dimensions of the level 0 image
dimensions = wsi.level_dimensions[level]

# get the entire level 0 image of the wsi
level_0_image = wsi.read_region((0, 0), level, dimensions)  

# if the image is RGBA, convert it to RGB
if level_0_image.mode == 'RGBA':
    level_0_image = level_0_image.convert('RGB')

print("Done getting the level 0 image.")
print(f"Time taken: {time.time() - start_time:.2f} seconds")

print("Downsampling the image by a factor of 2...")
start_time = time.time()
# downsample the image by a factor of 2
downsample_factor = 2
downsampled_image = level_0_image.resize((dimensions[0] // downsample_factor, dimensions[1] // downsample_factor))
print("Done downsampling the image.")
print(f"Time taken: {time.time() - start_time:.2f} seconds")