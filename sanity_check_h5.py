import h5py
import numpy as np

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


def track_unfilled_keys(h5_path):
    """
    Tracks which datasets in an HDF5 file contain entries that are still at their default value.

    Parameters:
    - h5_path (str): Path to the HDF5 file.

    Returns:
    - dict: A dictionary where the keys are dataset names and the values are booleans,
            indicating whether all entries in the dataset are still unfilled (True) or not (False).
    """
    default_value_tracker = {}

    with h5py.File(h5_path, "r") as f:
        for key in f.keys():
            dataset = f[key]

            # Check if the dataset is empty (all values are default)
            if isinstance(dataset[0], bytes):
                # Variable-length string default is empty byte string (b'')
                default_value = b""
                is_unfilled = np.all(dataset[:] == default_value)
            else:
                # For numeric types, default is zero
                default_value = 0
                is_unfilled = np.all(dataset[:] == default_value)

            # Add to the tracker
            default_value_tracker[key] = is_unfilled

    return default_value_tracker


print("Unfilled keys in slide h5:")
print(track_unfilled_keys(slide_h5_path))

print("Unfilled keys in heatmap h5:")
print(track_unfilled_keys(heatmap_h5_path))
