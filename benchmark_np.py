import time
import numpy as np
from PIL import Image

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

# downsample the image by a factor of 2 and record how long it takes
start_time = time.time()
slide_img_resized = slide_img.resize((width // 2, height // 2))
print(f"Time taken to resize image: {time.time() - start_time} seconds")
