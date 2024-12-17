import os
import ray
import h5py
import torch
import openslide
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from uni import load_model
from torch.utils.data import Dataset
from ray.exceptions import RayTaskError

patch_grid_csv = (
    "/home/dog/Documents/huong/analysis/visualization/website/mayo/K106022_coords.csv"
)
wsi_path = "/media/ssd2/huong/mayo_bbd/test_visual/process_img_list/K106022.svs"


def create_list_of_batches_from_list(list, batch_size):
    """
    This function creates a list of batches from a list.

    :param list: a list
    :param batch_size: the size of each batch
    :return: a list of batches

    >>> create_list_of_batches_from_list([1, 2, 3, 4, 5], 2)
    [[1, 2], [3, 4], [5]]
    >>> create_list_of_batches_from_list([1, 2, 3, 4, 5, 6], 3)
    [[1, 2, 3], [4, 5, 6]]
    >>> create_list_of_batches_from_list([], 3)
    []
    >>> create_list_of_batches_from_list([1, 2], 3)
    [[1, 2]]
    """

    list_of_batches = []

    for i in range(0, len(list), batch_size):
        batch = list[i : i + batch_size]
        list_of_batches.append(batch)

    return list_of_batches


def batching_tensor_stack(tensor_stack, batch_size):
    """
    The tensorstack has shape [num_samples, C, H, W]
    Batch the tensor stack into batches of size batch_size
    """

    batches = []

    for i in range(0, tensor_stack.shape[0], batch_size):
        batch = tensor_stack[i : i + batch_size]
        batches.append(batch)

    return batches


class SVSTileDataset(Dataset):
    def __init__(self, svs_path, csv_path, mpp=0.5, tile_size=224, transform=None):
        """
        Args:
            svs_path (str): Path to the SVS file (whole slide image).
            csv_path (str): Path to the CSV file with metadata and tile info.
            csv (pd.DataFrame): DataFrame with metadata and tile info.
            mpp (float): the mpp at which we want the patch to be extracted
            tile_size (int): the size of the tile to extract
            transform (callable, optional): Optional transform to be applied
                on a sample.
            svs (openslide.OpenSlide): OpenSlide object for the SVS file.
            level_0_mpp (float): the mpp at level 0 of the SVS file.
            downsampling_factor (float): the downsampling factor to get to the desired mpp.
            level_0_tile_size (int): the tile size at level 0 of the SVS file.
        """

        self.svs_path = svs_path
        self.csv_path = csv_path
        self.tile_size = tile_size
        self.transform = transform
        self.csv = pd.read_csv(csv_path)
        # only keep the rows where the include column is equal to True
        self.csv = self.csv[self.csv["include"] == True]
        self.mpp = mpp

        self.svs = openslide.OpenSlide(svs_path)

        self.level_0_mpp = (
            0.5  # float(self.svs.properties[openslide.PROPERTY_NAME_MPP_X]) TODO
        )

        # assert self.mpp > self.level_0_mpp, "mpp should be greater than the level 0 mpp" # Generally this must be checked
        self.downsampling_factor = 1  # self.mpp / self.level_0_mpp <<< generally we do not want to assume this TODO
        self.level_0_tile_size = int(tile_size * self.downsampling_factor)

    def __len__(self):
        # The length of the dataset is the number of rows in the metadata CSV
        return len(self.csv)

    def __getitem__(self, idx):
        # Get the metadata for the current index
        metadata_row = self.csv.iloc[idx]

        # Get the coordinates for cropping
        x = metadata_row["x"]
        y = metadata_row["y"]

        x_int, y_int = int(x), int(y)

        # Read the corresponding tile from the SVS file
        tile = self.svs.read_region(
            (x_int, y_int),
            level=0,
            size=(self.level_0_tile_size, self.level_0_tile_size),
        )
        tile = tile.convert("RGB")  # Convert to RGB if it's not already

        # # reshape the tile to the desired size # TODO this is in general a necessary step and can be a bottleneck
        # tile = tile.resize(
        #     (self.tile_size, self.tile_size)
        # )  # this the stage when the downsampling happens˝

        # Convert to numpy array for easier manipulation (e.g., transformations)
        tile = np.array(tile)

        # Apply the transformation if provided
        if self.transform:
            tile = self.transform(tile)

        # Convert to PyTorch tensor
        tile = torch.tensor(tile, dtype=torch.float32).permute(
            2, 0, 1
        )  # Change shape to CxHxW

        return tile  # TODO you might want to add some other metadata here to be tracked but these should not non-negligibly contribute to the overall runtime


@ray.remote
class TilingWorker:
    """
    === Attributes ===

    svs_path (str): path to the SVS file
    svs (openslide.OpenSlide): OpenSlide object for the SVS file
    """

    def __init__(self, svs_path):
        self.svs_path = svs_path
        self.svs = openslide.OpenSlide(svs_path)

    def async_tile(self, x, y, tile_size=224):
        """
        Get a tile from the SVS file at the given coordinates

        Args:
            x (int): x-coordinate of the tile
            y (int): y-coordinate of the tile
            tile_size (int): size of the tile

        Returns:
            np.ndarray: the tile
        """
        tile = self.svs.read_region((x, y), level=0, size=(tile_size, tile_size))

        return np.array(tile)

    def async_tile_batch(self, batch, tile_size=224, sub_batch_size=32):
        tiles = []

        for x, y in batch:
            tile = self.svs.read_region(
                (int(x), int(y)), level=0, size=(tile_size, tile_size)
            )

            # if RGBA convert to RGB
            if tile.mode == "RGBA":
                tile = tile.convert("RGB")

            tiles.append(tile)

        tensor_stack = torch.stack(
            [torch.tensor(np.array(tile)).permute(2, 0, 1) for tile in tiles]
        )

        tensor_batches = batching_tensor_stack(tensor_stack, sub_batch_size)

        return tensor_batches, len(batch)


num_tilers = 128
tiling_batch_size = 512

tiling_workers = [TilingWorker.remote(wsi_path) for _ in range(num_tilers)]

csv_path = (
    "/home/dog/Documents/huong/analysis/visualization/website/mayo/K106022_coords.csv"
)

# Load the CSV file
patch_grid = pd.read_csv(csv_path)

# only keep the rows where the include column is equal to True
patch_grid = patch_grid[patch_grid["include"] == True]

# get the x and y column as a list of (x,y) tuples
patch_grids = patch_grid[["x", "y"]].values.tolist()

print(f"Number of tiles: {len(patch_grid)}")

patch_grids_batches = create_list_of_batches_from_list(patch_grids, tiling_batch_size)

print(f"Number of batches: {len(patch_grids_batches)}")

tasks = {}
all_results = []
new_focus_regions = []

for i, batch in tqdm(
    enumerate(patch_grids_batches), desc="Tiling Tiles", total=len(patch_grids_batches)
):
    worker = tiling_workers[i % num_tilers]
    task = worker.async_tile_batch.remote(batch)
    tasks[task] = batch

with tqdm(total=len(patch_grids), desc="Extracting features") as pbar:
    while tasks:
        done_ids, _ = ray.wait(list(tasks.keys()))

        for done_id in done_ids:
            try:
                tensor_batches, update = ray.get(
                    done_id
                )  # this has dimension [batch_size, feature_size]

                all_results.extend(tensor_batches)

                pbar.update(update)
            except RayTaskError as e:
                print(f"Task for focus {tasks[done_id]} failed with error: {e}")

            del tasks[done_id]

print(len(all_results))
print(type(all_results[0]))
print(all_results[0].shape)

# import sys
# sys.exit()

# tile_batches = create_list_of_batches_from_list(all_results, 32)

# for i, batch in tqdm(
#     enumerate(tile_batches), desc="Stacking Tensors", total=len(tile_batches)
# ):

#     stack = torch.stack(
#         [torch.tensor(np.array(tile)).permute(2, 0, 1) for tile in batch]
#     )

# print("Creating the dataset...")
# # Create the dataset
# dataset = SVSTileDataset(
#     svs_path=wsi_path,
#     csv_path=patch_grid_csv,
#     mpp=0.5,
#     tile_size=224,
#     transform=None,
# )

# # print the length of the dataset
# print(f"Length of the dataset: {len(dataset)}")

# print("Load the model...")
# # Load the model
# model = load_model()
# # move the model to the GPU
# model.to("cuda")

# print("Getting the first sample...")
# # get the first sample and then print the type and shape of the tile
# sample = dataset[0]
# print(f"Type: {type(sample[0])}, Shape: {sample[0].shape}")


# dataloader = torch.utils.data.DataLoader(
#     dataset, batch_size=32, shuffle=True, num_workers=128
# )

# # get the first batch and print the shape
# batch = next(iter(dataloader))

# print(f"Batch shape: {batch.shape}")

# print(f"Moving the batch to the GPU...")
# # move the batch to the GPU
# batch = batch.to("cuda")

# print(f"Getting the features...")
# # get the features
# output = model(batch)

# print(f"Output shape: {output.shape}")
# print(f"Output: {output}")


# @ray.remote(num_gpus=1)
# class UNIFeatureExtractionWorker:
#     """Class for extracting features from tiles using UNI model
#     === Class Attributes ===
#     - model: UNIExtractor: the UNI model
#     """

#     def __init__(self):
#         """Initialize the UNIFeatureExtractionWorker"""
#         self.model = load_model()
#         self.model.to("cuda")

#     def async_extract_features(self, batch):
#         """Extract features from a batch of tiles
#         Args:
#             - batch: torch.Tensor: a batch of tiles
#         Returns:
#             - torch.Tensor: the features extracted from the tiles
#         """
#         # move the batch to the GPU
#         batch = batch.to("cuda")

#         # get the features
#         output = self.model(batch)

#         # delete teh batch from the GPU

#         # move the output to the CPU
#         output = output.to("cpu")

#         return output


# ray.shutdown()
# # ray.init(num_cpus=num_cpus, num_gpus=num_gpus)
# ray.init()

# num_feature_extractors = 8

# feature_extraction_workers = [
#     UNIFeatureExtractionWorker.remote() for _ in range(num_feature_extractors)
# ]

# tasks = {}
# all_results = []
# new_focus_regions = []

# for i, batch in tqdm(enumerate(dataloader), desc="Preparing tasks", total=len(dataset)):
#     worker = feature_extraction_workers[i % num_feature_extractors]
#     task = worker.async_extract_features.remote(batch)
#     tasks[task] = batch

# with tqdm(total=len(dataset), desc="Extracting features") as pbar:
#     while tasks:
#         done_ids, _ = ray.wait(list(tasks.keys()))

#         for done_id in done_ids:
#             try:
#                 batch_features = ray.get(
#                     done_id
#                 )  # this has dimension [batch_size, feature_size]

#                 all_results.append(batch_features)

#                 pbar.update(batch_features.shape[0])

#             except RayTaskError as e:
#                 print(f"Task for focus {tasks[done_id]} failed with error: {e}")

#             del tasks[done_id]

# # all features have dimension [batch_size, feature_size], concatenate them along the batch dimension to get [num_samples, feature_size]
# all_results = torch.cat(all_results, dim=0)
