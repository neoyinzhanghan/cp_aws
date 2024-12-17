import os
import ray
import h5py
import torch
import openslide
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from uni import load_model


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


patch_grid_csv = (
    "/home/dog/Documents/huong/analysis/visualization/website/mayo/K106022_coords.csv"
)
wsi_path = "/media/ssd2/huong/mayo_bbd/test_visual/process_img_list/K106022.svs"

print("Creating the dataset...")
# Create the dataset
dataset = SVSTileDataset(
    svs_path=wsi_path,
    csv_path=patch_grid_csv,
    mpp=0.5,
    tile_size=224,
    transform=None,
)

print("Load the model...")
# Load the model
model = load_model()
# move the model to the GPU
model.to("cuda")

print("Getting the first sample...")
# get the first sample and then print the type and shape of the tile
sample = dataset[0]
print(f"Type: {type(sample[0])}, Shape: {sample[0].shape}")


dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=128, shuffle=True, num_workers=128
)

# get the first batch and print the shape
batch = next(iter(dataloader))

print(f"Batch shape: {batch.shape}")

print(f"Moving the batch to the GPU...")
# move the batch to the GPU
batch = batch.to("cuda")

print(f"Getting the features...")
# get the features
output = model(batch)

print(f"Output shape: {output.shape}")
print(f"Output: {output}")
