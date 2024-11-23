import numpy as np

fg_np_path = "/media/hdd3/neo/viewer_sample_huong/website/foreground_mask.npy"
val_np_path = "/media/hdd3/neo/viewer_sample_huong/website/value_mask.npy"
slide_np_path = "/media/hdd3/neo/viewer_sample_huong/website/390359_mask.npy"

# Load the numpy arrays
fg_np = np.load(fg_np_path)
val_np = np.load(val_np_path)
slide_np = np.load(slide_np_path)

# print the shape of the numpy arrays
print(f"Foreground mask shape: {fg_np.shape}")
print(f"Value mask shape: {val_np.shape}")
print(f"Slide mask shape: {slide_np.shape}")
