import ray
import os
import io
import ray
import time
import h5py
import random
import base64
import shutil
import openslide
import numpy as np
from PIL import Image
from tqdm import tqdm


def image_to_jpeg_string(image):
    # Create an in-memory bytes buffer
    buffer = io.BytesIO()
    try:
        # Save the image in JPEG format to the buffer
        image.save(buffer, format="JPEG")
        jpeg_string = buffer.getvalue()  # Get the byte data
    finally:
        buffer.close()  # Explicitly close the buffer to free memory

    return jpeg_string


def jpeg_string_to_image(jpeg_string):
    # Create an in-memory bytes buffer from the byte string
    buffer = io.BytesIO(jpeg_string)

    # Open the image from the buffer and keep the buffer open
    image = Image.open(buffer)

    # Load the image data into memory so that it doesn't depend on the buffer anymore
    image.load()

    return image


def encode_image_to_base64(jpeg_string):
    return base64.b64encode(jpeg_string)


def decode_image_from_base64(encoded_string):
    return base64.b64decode(encoded_string)


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


def save_pyramid_to_h5(image_pyramid, h5_path):
    with h5py.File(h5_path, "w") as h5_file:
        for level, img_array in tqdm(image_pyramid.items(), desc="Saving to HDF5"):
            # Save each level as a dataset in the HDF5 file
            h5_file.create_dataset(name=str(level), data=img_array, compression="gzip")
    print(f"Image pyramid saved to {h5_path}")


@ray.remote
class WSICropManager:
    """
    A class representing a manager that crops WSIs.
    Each Manager object is assigned with a single CPU core and is responsible for cropping a subset of the coordinates from a given WSI.

    Attributes:
    wsi_path: str: The path to the WSI.
    wsi: openslide.OpenSlide: The WSI object.

    """

    def __init__(self, wsi_path) -> None:
        self.wsi_path = wsi_path
        self.wsi = None

    def open_slide(self):
        """Open the WSI."""
        self.wsi = openslide.OpenSlide(self.wsi_path)

    def close_slide(self):
        """Close the WSI."""
        self.wsi.close()
        self.wsi = None

    def get_level_0_dimensions(self):
        """Get dimensions of the slide at level 0."""
        if self.wsi is None:
            self.open_slide()
        return self.wsi.dimensions

    def get_level_N_dimensions(self, wsi_level):
        """Get dimensions of the slide at level N."""
        if self.wsi is None:
            self.open_slide()
        return self.wsi.level_dimensions[wsi_level]

    def get_tile_coordinate_level_pairs(self, tile_size=256, wsi_level=0):
        """Generate a list of coordinates_leve for 256x256 disjoint patches."""
        if self.wsi is None:
            self.open_slide()

        width, height = self.get_level_N_dimensions(wsi_level)
        coordinates = []

        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                # Ensure that the patch is within the image boundaries

                coordinates.append(
                    (
                        (x, y, min(x + tile_size, width), min(y + tile_size, height)),
                        wsi_level,
                    )
                )

        return coordinates

    def crop(self, coords, wsi_level=0):
        """Crop the WSI at the specified level of magnification."""
        if self.wsi is None:
            self.open_slide()

        coords_level_0 = (
            coords[0] * (2**wsi_level),
            coords[1] * (2**wsi_level),
            coords[2] * (2**wsi_level),
            coords[3] * (2**wsi_level),
        )

        image = self.wsi.read_region(
            (coords_level_0[0], coords_level_0[1]),
            wsi_level,
            (coords[2] - coords[0], coords[3] - coords[1]),
        )

        image = image.convert("RGB")
        return image

    def async_get_bma_focus_region_level_pair_batch(
        self, focus_region_coords_level_pairs, crop_size=256
    ):
        """Save a list of focus regions."""

        indices_to_jpeg = []
        for focus_region_coord_level_pair in focus_region_coords_level_pairs:
            focus_region_coord, wsi_level = focus_region_coord_level_pair

            image = self.crop(focus_region_coord, wsi_level=wsi_level)

            jpeg_string = image_to_jpeg_string(image)
            jpeg_string = encode_image_to_base64(jpeg_string)

            indices_level_jpeg = (
                focus_region_coord[0] // crop_size,
                focus_region_coord[1] // crop_size,
                wsi_level,
                jpeg_string,
            )

            indices_to_jpeg.append(indices_level_jpeg)

        return indices_to_jpeg
    

def get_tile_coordinate_level_pairs_level_0(image_width, image_height, tile_size=256):
    """Generate a list of coordinates_leve for tile_sizextile_size disjoint patches."""
    coordinates = []

    for y in range(0, image_height, tile_size):
        for x in range(0, image_width, tile_size):
            # Ensure that the patch is within the image boundaries

            coordinates.append(
                (
                    (x, y, min(x + tile_size, image_width), min(y + tile_size, image_height)),
                    0,
                )
            )

    return coordinates

def initialize_final_h5py_file(
    h5_path, image_width, image_height, num_levels=18, patch_size=256
):
    """
    Create an HDF5 file with a dataset that stores tiles, indexed by row and column.

    Parameters:
        h5_path (str): Path where the HDF5 file will be created.
        image_shape (tuple): Shape of the full image (height, width, channels).
        patch_size (int): The size of each image patch (default: 256).

    Raises:
        AssertionError: If the file already exists at h5_path.
    """
    if os.path.exists(h5_path):
        # delete the file
        os.remove(h5_path)

    # Create the HDF5 file and dataset
    with h5py.File(h5_path, "w") as f:
        # Create dataset with shape (num_tile_rows, num_tile_columns, patch_size, patch_size, 3)
        for level in range(num_levels + 1):
            level_image_height = image_height / (2 ** (num_levels - level))
            level_image_width = image_width / (2 ** (num_levels - level))

            dt = h5py.special_dtype(vlen=bytes)

            f.create_dataset(
                str(level),
                shape=(
                    max(level_image_width // patch_size + 1, 1),
                    max(level_image_height // patch_size + 1, 1),
                ),
                dtype=dt,
            )

        # also track the image width and height
        f.create_dataset(
            "level_0_width",
            shape=(1,),
            dtype="int",
        )

        f.create_dataset(
            "level_0_height",
            shape=(1,),
            dtype="int",
        )

        # also track the patch size
        f.create_dataset(
            "patch_size",
            shape=(1,),
            dtype="int",
        )

        # also track the number of levels
        f.create_dataset(
            "num_levels",
            shape=(1,),
            dtype="int",
        )

        # also track the number for overlap which is 0
        f.create_dataset(
            "overlap",
            shape=(1,),
            dtype="int",
        )

        f["level_0_width"][0] = image_width
        f["level_0_height"][0] = image_height
        f["patch_size"][0] = patch_size
        f["num_levels"][0] = num_levels
        f["overlap"][0] = 0

def initialize_final_h5py_file_and_tile_level_0(wsi_path, h5_path, num_levels=18, patch_size=256, batch_size=256, num_croppers=32):
    
    # get the level 0 dimensions as width, height of the wsi at level 0
    wsi = openslide.OpenSlide(wsi_path)
    image_width, image_height = wsi.dimensions

    initialize_final_h5py_file(h5_path, image_width, image_height, num_levels, patch_size)

    # get the tile coordinate level pairs for level 0
    tile_coordinate_level_pairs = get_tile_coordinate_level_pairs_level_0(
        image_width, image_height, tile_size=patch_size
    )

    # create a list of batches
    list_of_batches = create_list_of_batches_from_list(tile_coordinate_level_pairs, batch_size)

    task_managers = [WSICropManager.remote(wsi_path) for _ in range(num_croppers)]

    tasks = {}

    for i, batch in enumerate(list_of_batches):
        manager = task_managers[i % num_croppers]
        task = manager.async_get_bma_focus_region_level_pair_batch.remote(
            batch, crop_size=patch_size
        )
        tasks[task] = batch
    with h5py.File(h5_path, "a") as f:
        with tqdm(
            total=len(tile_coordinate_level_pairs), desc="Cropping focus regions"
        ) as pbar:
            while tasks:
                done_ids, _ = ray.wait(list(tasks.keys()))

                for done_id in done_ids:
                    try:
                        batch = ray.get(done_id)
                        for indices_jpeg in batch:
                            x, y, wsi_level, jpeg_string = indices_jpeg
                            level = int(18 - wsi_level)
                            f[str(level)][x, y] = jpeg_string
                            # print(f"Saved patch at level: {level}, x: {x}, y: {y}")
                            # print(f"jpeg_string: {jpeg_string}")

                        pbar.update(len(batch))

                    except ray.exceptions.RayTaskError as e:
                        print(f"Task for batch {tasks[done_id]} failed with error: {e}")

                    del tasks[done_id]

if __name__ == "__main__":
    wsi_path = '/media/ssd2/neo/cp_aws_playground/23.CFNA.113 A1 H&E _171848.svs'
    h5_path = '/media/ssd2/neo/cp_aws_playground/23.CFNA.113 A1 H&E _171848.h5'

    if os.path.exists(h5_path):
        # delete the file
        os.remove(h5_path)

    initialize_final_h5py_file_and_tile_level_0(wsi_path, h5_path, num_levels=18, patch_size=256, batch_size=1024, num_croppers=200)

    