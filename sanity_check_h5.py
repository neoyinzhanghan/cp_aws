import os
import sys
import h5py
import numpy as np
from dzsave_h5 import retrieve_tile_h5

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


def track_tile_errors(h5_path):
    """
    Tracks the number of errors thrown by `retrieve_tile_h5` for each integer level in the HDF5 file.

    Parameters:
    - h5_path (str): Path to the HDF5 file.
    - retrieve_tile_h5 (callable): Function to retrieve a tile from the HDF5 file.
                                   Should accept (h5_path, level, row, col) as parameters.

    Returns:
    - dict: A dictionary where the keys are levels (integer strings) and the values are the number of errors
            encountered for that level.
    """
    error_tracker = {}

    with h5py.File(h5_path, "r") as f:
        # Iterate over all integer string keys
        for level in f.keys():
            if level.isdigit():  # Check if the key is an integer string
                level_dataset = f[level]
                num_rows, num_cols = level_dataset.shape  # Get the shape of the dataset
                error_count = 0

                # Iterate over all rows and columns
                for row in range(num_rows):
                    for col in range(num_cols):
                        try:
                            # Call the retrieve_tile_h5 function
                            retrieve_tile_h5(h5_path, level, row, col)
                        except Exception as e:
                            # Increment error count if an exception occurs
                            error_count += 1

                # Record the error count for the level
                error_tracker[level] = error_count

    return error_tracker


# Suppress all prints and logs
def suppress_logs(func, *args, **kwargs):
    """
    Suppresses all console outputs during the execution of a function.

    Parameters:
    - func (callable): The function to execute.
    - *args: Positional arguments for the function.
    - **kwargs: Keyword arguments for the function.

    Returns:
    - The return value of the function.
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull  # Redirect stdout
            sys.stderr = devnull  # Redirect stderr
            return func(*args, **kwargs)
        finally:
            sys.stdout = old_stdout  # Restore stdout
            sys.stderr = old_stderr  # Restore stderr


print("Unfilled keys in slide h5:")
tracker = suppress_logs(track_tile_errors, slide_h5_path, retrieve_tile_h5)
print(tracker)

print("Unfilled keys in heatmap h5:")
tracker = suppress_logs(track_tile_errors, heatmap_h5_path, retrieve_tile_h5)
print(tracker)
