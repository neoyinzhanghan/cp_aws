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
print(f"Time taken to convert numpy array to image: {time.time() - start_time} seconds")
