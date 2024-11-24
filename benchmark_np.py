import time
import numpy as np
from PIL import Image
from tqdm import tqdm

slide_np_path = "/media/hdd3/neo/viewer_sample_huong/website/390359_mask.npy"
save_dir = "/media/hdd3/neo/viewer_sample_huong/website/test_tmp_dir"

# Load the numpy arrays
start_time = time.time()
slide_np = np.load(slide_np_path)
print(f"Time taken to load numpy array: {time.time() - start_time} seconds")

# convert the numpy array to an image
start_time = time.time()
slide_img = Image.fromarray(slide_np)
# make sure the image is in RGB mode
slide_img = slide_img.convert("RGB")
print(f"Time taken to convert numpy array to image: {time.time() - start_time} seconds")

height, width = slide_img.size

# create an image pyramid with 18 levels
start_time = time.time()
num_levels = 18
image_pyramid_dict = {}
current_img = slide_img
for i in tqdm(range(num_levels + 1), desc="Creating image pyramid"):
    level = num_levels - i

    current_img = current_img.resize((width // 2**level, height // 2**level))
    image_pyramid_dict[level] = current_img
print(f"Time taken to create image pyramid: {time.time() - start_time} seconds")
