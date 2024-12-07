import io
import os
import ray
import time
import h5py
import base64

import openslide
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image


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


def crop_wsi_images_all_levels(
    wsi_path,
    h5_path,
    region_cropping_batch_size,
    crop_size=256,
    verbose=True,
    num_cpus=32,
):
    num_croppers = num_cpus  # Number of croppers is the same as num_cpus

    wsi = openslide.OpenSlide(wsi_path)
    level_count = wsi.level_count
    if verbose:
        print("Initializing WSICropManager")

    manager = WSICropManager.remote(wsi_path)

    # Get all the coordinates for 256x256 patches
    focus_regions_coordinates = []

    for level in range(level_count):
        focus_regions_coordinates.extend(
            ray.get(
                manager.get_tile_coordinate_level_pairs.remote(
                    tile_size=crop_size, wsi_level=level
                )
            )
        )
    list_of_batches = create_list_of_batches_from_list(
        focus_regions_coordinates, region_cropping_batch_size
    )

    task_managers = [WSICropManager.remote(wsi_path) for _ in range(num_croppers)]

    tasks = {}

    for i, batch in enumerate(list_of_batches):
        manager = task_managers[i % num_croppers]
        task = manager.async_get_bma_focus_region_level_pair_batch.remote(
            batch, crop_size=crop_size
        )
        tasks[task] = batch
    with h5py.File(h5_path, "a") as f:
        with tqdm(
            total=len(focus_regions_coordinates), desc="Cropping focus regions"
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


def get_depth_from_0_to_N(wsi_path, h5_path, tile_size=256):

    wsi = openslide.OpenSlide(wsi_path)
    level_count = wsi.level_count

    assert level_count <= 18, "The slide has more than 18 levels"

    level_top_dimensions = wsi.level_dimensions[level_count - 1]
    image = wsi.read_region((0, 0), level_count - 1, level_top_dimensions)
    image = image.convert("RGB")

    current_image = image
    for depth in range(18 - level_count, -1, -1):
        # downsample the image by a factor of 2
        current_image = image.resize(
            (
                int(max(image.width // (2 ** ((18 - level_count + 1) - depth)), 1)),
                int(max(image.height // (2 ** ((18 - level_count + 1) - depth)), 1)),
            )
        )
        # crop 256x256 patches from the downsampled image (don't overlap, dont leave out any boundary patches)
        for y in range(0, current_image.height, tile_size):
            for x in range(0, current_image.width, tile_size):
                # Calculate the right and bottom coordinates ensuring they are within the image boundaries
                right = min(x + tile_size, current_image.width)
                bottom = min(y + tile_size, current_image.height)

                # Crop the patch from the image starting at (x, y) to (right, bottom)
                patch = current_image.crop((x, y, right, bottom))

                # make sure patch is in RGB mode and a PIL image
                patch = patch.convert("RGB")

                # Save the patch to the h5 file
                with h5py.File(h5_path, "a") as f:
                    jpeg_string = image_to_jpeg_string(patch)
                    jpeg_string = encode_image_to_base64(jpeg_string)
                    try:
                        f[str(depth)][
                            int(x // tile_size), int(y // tile_size)
                        ] = jpeg_string
                    except Exception as e:
                        print(
                            f"Error: {e} occurred while saving patch at level: {depth}, x: {x}, y: {y} to {h5_path}"
                        )


def dzsave_h5(
    wsi_path,
    h5_path,
    tile_size=256,
    num_cpus=32,
    region_cropping_batch_size=256,
):
    """
    Create a DeepZoom image pyramid from a WSI.
    """

    wsi = openslide.OpenSlide(wsi_path)
    width, height = wsi.dimensions
    image_width, image_height = wsi.dimensions

    initialize_final_h5py_file(
        h5_path,
        image_width=image_width,
        image_height=image_height,
        patch_size=tile_size,
    )

    print(f"Width: {width}, Height: {height}")

    starttime = time.time()

    print("Cropping from WSI using native levels")
    crop_wsi_images_all_levels(
        wsi_path,
        h5_path,
        region_cropping_batch_size=region_cropping_batch_size,
        crop_size=tile_size,
        num_cpus=num_cpus,
    )

    print("Cropping Lower Resolution Levels")
    get_depth_from_0_to_N(wsi_path, h5_path, tile_size=tile_size)

    time_taken = time.time() - starttime

    return time_taken


def retrieve_tile_h5(h5_path, level, row, col):
    with h5py.File(h5_path, "r") as f:
        try:
            jpeg_string = f[str(level)][row, col]
            jpeg_string = decode_image_from_base64(jpeg_string)
            image = jpeg_string_to_image(jpeg_string)

        except Exception as e:
            print(
                f"Error: {e} occurred while retrieving tile at level: {level}, row: {row}, col: {col} from {h5_path}"
            )
            jpeg_string = f[str(level)][row, col]
            print(f"jpeg_string: {jpeg_string}")
            jpeg_string = decode_image_from_base64(jpeg_string)
            print(f"jpeg_string base 64 decoded: {jpeg_string}")
            raise e
        return image


import openslide
from PIL import Image
import pyvips
import numpy as np
import time


def dyadically_reorganize_svs_levels(input_svs_path, output_svs_path):
    """
    Process an SVS file to reorganize its levels to have a consistent downsampling factor of 2
    and save the result as a new SVS file with preserved metadata.

    Args:
        input_svs_path (str): Path to the input SVS file.
        output_svs_path (str): Path to save the output SVS file.
    """
    start_time = time.time()
    try:
        print(f"Starting processing for: {input_svs_path}")

        # Open the original SVS file
        print("Opening the input SVS file...")
        slide = openslide.OpenSlide(input_svs_path)

        # Retrieve metadata
        print("Retrieving metadata...")
        mpp_x = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_X, 0))
        mpp_y = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_Y, 0))
        level_dimensions = slide.level_dimensions
        level_downsamples = slide.level_downsamples

        level_downsamples = [int(downsample) for downsample in level_downsamples]

        if mpp_x == 0 or mpp_y == 0:
            print("Warning: MPP values not found in the input file.")

        print(f"Metadata retrieved: MPP X: {mpp_x}, MPP Y: {mpp_y}")
        print(f"Level dimensions: {level_dimensions}")
        print(f"Level downsamples: {level_downsamples}")

        # Read the base level image
        base_level = 0
        print(f"Reading base level: {base_level}")
        base_image = slide.read_region((0, 0), base_level, level_dimensions[base_level])
        base_image = base_image.convert("RGB")
        print(f"Base level dimensions: {level_dimensions[base_level]}")

        # Generate pyramid levels with downsampling factor of 2
        print("Generating pyramid levels with a downsampling factor of 2...")
        pyramid = [base_image]
        current_level = 0
        while pyramid[-1].size[0] > 1 and pyramid[-1].size[1] > 1:
            print(f"Processing level {current_level + 1}...")
            next_level = pyramid[-1].resize(
                (pyramid[-1].size[0] // 2, pyramid[-1].size[1] // 2), Image.LANCZOS
            )
            print(f"Level {current_level + 1} dimensions: {next_level.size}")
            pyramid.append(next_level)
            current_level += 1

        print(f"Generated {len(pyramid)} pyramid levels.")

        # Convert PIL images to pyvips images
        print("Converting pyramid levels to pyvips images...")
        vips_images = []
        for idx, img in enumerate(pyramid):
            print(f"Converting level {idx}...")
            vips_image = pyvips.Image.new_from_array(np.array(img))
            vips_images.append(vips_image)

        # Merge the images into a single pyvips pyramid
        print("Merging all levels into a single pyvips pyramid...")
        vips_pyramid = vips_images[0]
        for idx, level in enumerate(vips_images[1:], start=1):
            print(f"Merging level {idx}...")
            vips_pyramid = vips_pyramid.insert(level, 0, 0)

        # Add metadata to the new SVS file
        print("Adding metadata to the new SVS file...")
        metadata_description = {
            "openslide.mpp-x": str(mpp_x),
            "openslide.mpp-y": str(mpp_y),
            "openslide.level-count": str(len(pyramid)),
        }
        for level, dimensions in enumerate(level_dimensions):
            metadata_description[f"openslide.level[{level}].width"] = str(dimensions[0])
            metadata_description[f"openslide.level[{level}].height"] = str(
                dimensions[1]
            )
            metadata_description[f"openslide.level[{level}].downsample"] = str(
                level_downsamples[level]
            )

        # Create a metadata string for ImageDescription
        image_description = "\n".join(
            f"{k}: {v}" for k, v in metadata_description.items()
        )
        print("Metadata string created for ImageDescription.")

        # Save the pyramid as a new SVS file
        print(f"Saving the new SVS file to: {output_svs_path}")
        vips_pyramid.tiffsave(
            output_svs_path,
            tile=True,
            pyramid=True,
            compression="jpeg",
            tile_width=256,
            tile_height=256,
            bigtiff=True,
            properties=False,  # Disable automatic properties writing
            xres=mpp_x,  # Horizontal resolution
            yres=mpp_y,  # Vertical resolution
        )
        print(f"Reorganized SVS saved successfully to: {output_svs_path}")

    except Exception as e:
        print(f"Error processing {input_svs_path}: {e}")
        raise e

    finally:
        elapsed_time = time.time() - start_time
        print(
            f"Time taken to dyadically reorganize SVS levels: {elapsed_time:.2f} seconds"
        )


if __name__ == "__main__":
    svs_path = "/media/hdd3/neo/viewer_sample_huong/390359.svs"
    new_svs_path = "/media/hdd3/neo/viewer_sample_huong/390359_dyadic.svs"
    h5_path = "/media/hdd3/neo/viewer_sample_huong/390359.h5"

    dyadically_reorganize_svs_levels(
        svs_path,
        new_svs_path,
    )

    # if the h5 file already exists, delete it
    if os.path.exists(h5_path):
        os.remove(h5_path)

    dzsave_h5(
        new_svs_path,
        h5_path,
        tile_size=256,
        num_cpus=32,
        region_cropping_batch_size=256,
    )

    start_time = time.time()

    print(
        f"Time taken to dyadically reorganize SVS levels: {time.time() - start_time:.2f} seconds"
    )

    # image = retrieve_tile_h5(h5_path, 18, 93, 81)

    # # save the image at /media/hdd3/neo/test.jpeg
    # image.save("/media/hdd3/neo/test_convert.jpeg")
