import h5py

slide_h5_path = "/home/dog/Documents/neo/cp-lab-wsi-upload/wsi-and-heatmaps/390359.h5"

# try to get the level_0_width and level_0_height
with h5py.File(slide_h5_path, "r") as f:
    level_0_width = f["level_0_width"][()]
    level_0_height = f["level_0_height"][()]

print(level_0_width, level_0_height)