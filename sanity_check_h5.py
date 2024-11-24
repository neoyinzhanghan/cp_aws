import h5py

slide_h5_path = "/media/hdd3/neo/viewer_sample_huong/390359.h5"
heatmap_h5_path = "/media/hdd3/neo/viewer_sample_huong/390359_heatmap.h5"

# Load the h5 files
slide_h5 = h5py.File(slide_h5_path, "r")
heatmap_h5 = h5py.File(heatmap_h5_path, "r")

# print the keys of the h5 files
print(f"Slide h5 keys: {list(slide_h5.keys())}")
print(f"Heatmap h5 keys: {list(heatmap_h5.keys())}")

# map the key to the dimensions
for key in slide_h5.keys():
    print(f"Slide h5 key: {key}, dimensions: {slide_h5[key].shape}")

for key in heatmap_h5.keys():
    print(f"Heatmap h5 key: {key}, dimensions: {heatmap_h5[key].shape}")
