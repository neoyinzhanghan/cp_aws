import time
import openslide
from PIL import Image
from tqdm import tqdm

wsi_path = '/media/ssd2/neo/cp_aws_playground/23.CFNA.113 A1 H&E _171848.svs'

# print("Getting the level 0 image of the WSI...")
# start_time = time.time()
# # get the entire level 0 image of the wsi
# wsi = openslide.OpenSlide(wsi_path)
# level = 0

# # get the dimensions of the level 0 image
# dimensions = wsi.level_dimensions[level]

# # get the entire level 0 image of the wsi
# level_0_image = wsi.read_region((0, 0), level, dimensions)  

# # if the image is RGBA, convert it to RGB
# if level_0_image.mode == 'RGBA':
#     level_0_image = level_0_image.convert('RGB')

# print("Done getting the level 0 image.")
# print(f"Time taken: {time.time() - start_time:.2f} seconds")

# print("Downsampling the image by a factor of 2...")
# start_time = time.time()
# # downsample the image by a factor of 2
# downsample_factor = 2
# downsampled_image = level_0_image.resize((dimensions[0] // downsample_factor, dimensions[1] // downsample_factor))
# print("Done downsampling the image.")
# print(f"Time taken: {time.time() - start_time:.2f} seconds")

def create_image_pyramid_dct(level_0_image, downsample_factor=2, num_levels=18):
    """
    Create an image pyramid using a dictionary to store the images at each level.
    
    Args:
        level_0_image (PIL.Image): The level 0 image of the WSI.
        downsample_factor (int): The downsample factor for each level.
        num_levels (int): The number of levels in the image pyramid.
        
    Returns:
        dict: A dictionary containing the images at each level of the image pyramid.
    """

    start_time = time.time()
    image_pyramid = {}
    image_pyramid[0] = level_0_image
    current_image = level_0_image
    for level in tqdm(range(num_levels-1, -1, -1), desc="Creating image pyramid"):
        current_image = current_image.resize((max(current_image.width // downsample_factor, 1), max(current_image.height // downsample_factor, 1)))
        image_pyramid[level] = current_image
    
    print(f"Time taken to create image pyramid: {time.time() - start_time:.2f} seconds")

    return image_pyramid

if __name__ == "__main__":

    print("Getting the level 0 image of the WSI...")
    start_time = time.time()

    wsi = openslide.OpenSlide(wsi_path)
    level = 0

    # Get the dimensions of the level 0 image
    dimensions = wsi.level_dimensions[level]

    # Get the entire level 0 image of the WSI
    level_0_image = wsi.read_region((0, 0), level, dimensions)

    # If the image is RGBA, convert it to RGB
    if level_0_image.mode == 'RGBA':
        level_0_image = level_0_image.convert('RGB')

    print("Done getting the level 0 image.")
    print(f"Time taken: {time.time() - start_time:.2f} seconds")

    print("Creating the image pyramid...")
    start_time = time.time()    

    image_pyramid = create_image_pyramid_dct(level_0_image, downsample_factor=2, num_levels=18)

    print("Done creating the image pyramid.")
    print(f"Time taken: {time.time() - start_time:.2f} seconds")
    