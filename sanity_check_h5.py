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

# print the level_0_height and level_0_width
level_0_height = slide_h5["level_0_height"][()]
level_0_width = slide_h5["level_0_width"][()]
patch_size = slide_h5["patch_size"][()]

print(
    f"Slide Level 0 height: {level_0_height}, Level 0 width: {level_0_width}, Patch size: {patch_size}"
)

# do the same for the heatmap
level_0_height = heatmap_h5["level_0_height"][()]
level_0_width = heatmap_h5["level_0_width"][()]
patch_size = heatmap_h5["patch_size"][()]

print(
    f"Heatmap Level 0 height: {level_0_height}, Heatmap Level 0 width: {level_0_width}, Heatmap Patch size: {patch_size}"
)
