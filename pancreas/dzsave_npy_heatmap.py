import io
import os
import ray
import h5py
import base64
import numpy as np
from PIL import Image
from tqdm import tqdm

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



def dzsave_npy_heatmap(wsi_h5_path, heatmap_h5_path, npy_path):

    # open the WSI h5 file and extract the following information
    # level_0_width, level_0_height, patch_size, num_levels, overlap
    with h5py.File(wsi_h5_path, "r") as f:
        level_0_width = f["level_0_width"][0]
        level_0_height = f["level_0_height"][0]
        patch_size = f["patch_size"][0]
        num_levels = f["num_levels"][0] 
        overlap = f["overlap"][0] # this is an argument that we dont really use here
    
    # create a new h5 file to store the heatmap
    with h5py.File(heatmap_h5_path, "w") as f:
        for level in range(num_levels):
            # open the heatmap h5 file and extract the heatmap at each level
            with h5py.File(wsi_h5_path, "r") as g:
                wsi_lvl = g[str(level)][:]

                # create a dataset in the new h5 file to store the heatmap
                dt = h5py.special_dtype(vlen=bytes)

                # exact same shape as heatmap
                f.create_dataset(
                    str(level),
                    shape=(
                        wsi_lvl.shape[0],
                        wsi_lvl.shape[1],
                    ),
                    dtype=dt,
                )
        
        # create datasets to store the level 0 width and height and the patch size and num levels and overlap
        f.create_dataset("level_0_width", shape=(1,), dtype="int")
        f.create_dataset("level_0_height", shape=(1,), dtype="int")
        f.create_dataset("patch_size", shape=(1,), dtype="int")
        f.create_dataset("num_levels", shape=(1,), dtype="int")
        f.create_dataset("overlap", shape=(1,), dtype="int")

        # store the values
        f["level_0_width"][0] = level_0_width
        f["level_0_height"][0] = level_0_height
        f["patch_size"][0] = patch_size
        f["num_levels"][0] = num_levels
        f["overlap"][0] = overlap

    store_top_level_tiles(npy_path, heatmap_h5_path)

    # pyramid = create_image_pyramid_dct_from_numpy(...)


@ray.remote
class NPYCropManager:
    """
    A class representing a manager that crops WSIs.
    Each Manager object is assigned with a single CPU core and is responsible for cropping a subset of the coordinates from a given WSI.

    Attributes:
    npy_heatmap: numpy array representing the heatmap
    level_0_width: width of the level 0 image
    level_0_height: height of the level 0 image
    npy_width: width of the numpy array
    npy_height: height of the numpy array
    num_levels: number of levels in the image pyramid
    patch_size: size of the crop
    downsample_factor_width: downsample factor for the width
    downsample_factor_height: downsample factor for the height

    """

    def __init__(self, npy_heatmap_obj_ref, level_0_width, level_0_height, patch_size=256, num_levels=19) -> None:
        self.npy_heatmap = npy_heatmap_obj_ref

        npy_height, npy_width, _ = self.npy_heatmap.shape

        # assert that the npy dimensions are <= level 0 dimensions
        assert npy_width <= level_0_width, "Numpy width is greater than level 0 width"
        assert npy_height <= level_0_height, "Numpy height is greater than level 0 height"

        self.level_0_width = level_0_width
        self.level_0_height = level_0_height

        self.npy_width = npy_width
        self.npy_height = npy_height

        self.num_levels = num_levels
        downsample_factor_width = level_0_width / npy_width
        downsample_factor_height = level_0_height / npy_height

        self.patch_size = patch_size
        self.downsample_factor_width = downsample_factor_width
        self.downsample_factor_height = downsample_factor_height

        assert abs(downsample_factor_width - downsample_factor_height) <= 0.1, "Downsample factors are not equal up to a margin of 0.1"        

    def async_get_bma_focus_region_level_pair_batch(
        self, focus_region_coords_level_pairs
    ):
        """Save a list of focus regions."""

        indices_to_jpeg = []
        for focus_region_coord_level_pair in focus_region_coords_level_pairs:
            focus_region_coord, dzsave_level = focus_region_coord_level_pair

            assert dzsave_level == self.num_levels - 1, "dzsave_level is not the last level, this should indicate a grave error"

            subsampled_focus_region_coord = (
                focus_region_coord[0] // self.downsample_factor_width,
                focus_region_coord[1] // self.downsample_factor_height,
                focus_region_coord[2] // self.downsample_factor_width,
                focus_region_coord[3] // self.downsample_factor_height,
            )

            # crop the numpy array (but note that the x and y are swapped)
            np_image = self.npy_heatmap[
                subsampled_focus_region_coord[1] : subsampled_focus_region_coord[3],
                subsampled_focus_region_coord[0] : subsampled_focus_region_coord[2],
                :,
            ]

            # create a PIL image from the numpy array
            downsampled_image = Image.fromarray(np_image)

            # upsample the image to self.patch_size
            image = downsampled_image.resize(
                (self.patch_size, self.patch_size), Image.BICUBIC
            )

            # if the image is RGBA, convert it to RGB
            if image.mode == 'RGBA':
                raise ValueError("Image is RGBA, this should not happen")

            jpeg_string = image_to_jpeg_string(image)
            jpeg_string = encode_image_to_base64(jpeg_string)

            indices_level_jpeg = (
                focus_region_coord[0] // self.patch_size,
                focus_region_coord[1] // self.patch_size,
                dzsave_level,
                jpeg_string,
            )

            indices_to_jpeg.append(indices_level_jpeg)

        return indices_to_jpeg

def get_tile_coordinate_level_pairs_npy_top_level(image_width, image_height, patch_size=256, num_levels=19):
    """
    Get the tile coordinate level pairs for the top level of the image pyramid.

    Args:
        image_width (int): The width of the image.
        image_height (int): The height of the image.
        patch_size (int): The size of the patch.
        num_levels (int): The number of levels in the image pyramid.

    Returns:
        list: A list of tuples containing the tile coordinates and the level.
    """

    tile_coordinate_level_pairs = []
    for y in range(0, image_height, patch_size):
        for x in range(0, image_width, patch_size):
            tile_coordinate_level_pairs.append(((x, y, min(x + patch_size, image_width), min(y + patch_size, image_height)), num_levels - 1))   
    return tile_coordinate_level_pairs  

def store_top_level_tiles(npy_path, heatmap_h5_path, batch_size=1024, num_croppers=32):
    # open the numpy array
    heatmap = np.load(npy_path)
    heatmap_ref = ray.put(heatmap)

    # get the level 0 width and height from the heatmap h5 file
    with h5py.File(heatmap_h5_path, "r") as f:
        level_0_width = f["level_0_width"][0]
        level_0_height = f["level_0_height"][0]
        patch_size = f["patch_size"][0]
        num_levels = f["num_levels"][0]

    tile_coordinate_level_pairs = get_tile_coordinate_level_pairs_npy_top_level(
        level_0_width, level_0_height, patch_size=patch_size, num_levels=num_levels
    )

    # create a crop manager
    task_managers = [NPYCropManager.remote(heatmap_ref, level_0_width, level_0_height, patch_size=patch_size, num_levels=num_levels) for _ in range(num_croppers)]

    # create a list of batches
    list_of_batches = create_list_of_batches_from_list(tile_coordinate_level_pairs, batch_size)

    tasks = {}

    for i, batch in enumerate(list_of_batches):
        manager = task_managers[i % num_croppers]
        task = manager.async_get_bma_focus_region_level_pair_batch.remote(
            batch
        )
        tasks[task] = batch
    with h5py.File(heatmap_h5_path, "a") as f:
        with tqdm(
            total=len(tile_coordinate_level_pairs), desc="Cropping focus regions"
        ) as pbar:
            while tasks:
                done_ids, _ = ray.wait(list(tasks.keys()))

                for done_id in done_ids:
                    try:
                        batch = ray.get(done_id)
                        for indices_jpeg in batch:
                            x, y, dzsave_level, jpeg_string = indices_jpeg
                            try:
                                f[str(dzsave_level)][x, y] = jpeg_string
                            except Exception as e:
                                print(f"Error saving patch at level: {dzsave_level}, x: {x}, y: {y}, error: {e}")
                                raise e
                            # print(f"Saved patch at level: {level}, x: {x}, y: {y}")
                            # print(f"jpeg_string: {jpeg_string}")

                        pbar.update(len(batch))

                    except ray.exceptions.RayTaskError as e:
                        print(f"Task for batch {tasks[done_id]} failed with error: {e}")

                    del tasks[done_id]

    

# def create_image_pyramid_dct_from_numpy(...):
#     pass 

if __name__ == "__main__":
    wsi_path = '/media/ssd2/neo/cp_aws_playground/23.CFNA.113 A1 H&E _171848.svs'
    h5_path = '/media/ssd2/neo/cp_aws_playground/23.CFNA.113 A1 H&E _171848.h5' 
    npy_path = "/media/ssd2/neo/cp_aws_playground/23.CFNA.113 A1 H&E _171848_rainbow_heatmap_mask.npy"
    heatmap_h5_path = '/media/ssd2/neo/cp_aws_playground/23.CFNA.113 A1 H&E _171848_rainbow_heatmap.h5'

    if os.path.exists(heatmap_h5_path):
        # delete the file
        os.remove(heatmap_h5_path)

    dzsave_npy_heatmap(h5_path, heatmap_h5_path, npy_path)