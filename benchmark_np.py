import os
import io
import ray
import time
import h5py
import random
import base64
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


@ray.remote
class PILPyramidCropManager:
    """
    A class representing a manager that crops PIL.
    Each Manager object is assigned with a single CPU core and is responsible for cropping a subset of the coordinates from a given PIL.

    Attributes:
    pil_pyramid: a dictionary of PIL representing the image pyramid.
    """

    def __init__(self, pil_pyramid) -> None:
        self.pil_pyramid = pil_pyramid

    def get_level_N_dimensions(self, dz_level):
        """Get dimensions of the slide at level N."""

        return self.pil_pyramid[dz_level].size

    def crop(self, coords, dz_level=0):
        """Crop the numpy array at the specified level of magnification."""

        image = self.pil_pyramid[dz_level].crop(coords)

        return image

    def async_get_jpeg_string_batch(self, coord_level_pair_batch):
        """Save a list of focus regions."""

        indices_to_jpeg = []
        for coord_level_pair in coord_level_pair_batch:
            coord, dz_level = coord_level_pair

            image = self.crop(coord, dz_level)

            jpeg_string = image_to_jpeg_string(image)
            jpeg_string = encode_image_to_base64(jpeg_string)

            indices_level_jpeg = (coord[0], coord[1], dz_level, jpeg_string)

            indices_to_jpeg.append(indices_level_jpeg)

        return indices_to_jpeg


def get_tile_coordinate_level_pairs(pil_pyramid, tile_size=256):
    """Generate a list of coordinates_leve for 256x256 disjoint patches."""

    coordinates = []
    for dz_level in pil_pyramid.keys():
        width, height = pil_pyramid[dz_level].size

        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                # Ensure that the patch is within the image boundaries

                coordinates.append(
                    (
                        (
                            x,
                            y,
                            min(x + tile_size, width),
                            min(y + tile_size, height),
                        ),
                        dz_level,
                    )
                )

    return coordinates


@ray.remote
class H5PyramidCropManager:
    """
    A class representing a manager that crops PIL.
    Each Manager object is assigned with a single CPU core and is responsible for cropping a subset of the coordinates from a given PIL.

    Attributes:
    h5_pyramid_path: path to the h5 file representing the image pyramid.
    f: h5py.File: the h5 file object.
    """

    def __init__(self, h5_pyramid_path) -> None:
        self.h5_pyramid_path = h5_pyramid_path
        self.f = None

    def open_h5_file(self):
        """Open the h5 file."""
        self.f = h5py.File(self.h5_pyramid_path, "r")

    def close_h5_file(self):
        """Close the h5 file."""
        self.f.close()
        self.f = None

    def get_level_N_dimensions(self, dz_level):
        """Get dimensions of the slide at level N."""

        height, width, _ = self.f[str(dz_level)].shape

        return width, height

    def crop(self, coords, dz_level=0):
        """Crop the numpy array at the specified level of magnification.
        coords is TL_x, TL_y, BR_x, BR_y format
        """

        image = self.f[str(dz_level)][coords[1] : coords[3], coords[0] : coords[2], :]

        # convert the image to PIL image
        image = Image.fromarray(image)

        return image

    def async_get_jpeg_string_batch(self, coord_level_pair_batch):
        """Save a list of focus regions."""

        indices_to_jpeg = []
        for coord_level_pair in coord_level_pair_batch:
            coord, dz_level = coord_level_pair

            image = self.crop(coord, dz_level)

            jpeg_string = image_to_jpeg_string(image)
            jpeg_string = encode_image_to_base64(jpeg_string)

            indices_level_jpeg = (coord[0], coord[1], dz_level, jpeg_string)

            indices_to_jpeg.append(indices_level_jpeg)

        return indices_to_jpeg


def get_tile_coordinate_level_pairs(pil_pyramid, tile_size=256):
    """Generate a list of coordinates_leve for 256x256 disjoint patches."""

    coordinates = []
    for dz_level in pil_pyramid.keys():
        width, height = pil_pyramid[dz_level].size

        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                # Ensure that the patch is within the image boundaries

                coordinates.append(
                    (
                        (
                            x,
                            y,
                            min(x + tile_size, width),
                            min(y + tile_size, height),
                        ),
                        dz_level,
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


def dzsave_h5_np(
    np_path,
    h5_path,
    tile_size=256,
    num_cpus=32,
    region_cropping_batch_size=256,
):
    # Load the numpy array, turn into pil image and then create the image pyramid
    start_time = time.time()
    np_array = np.load(np_path)
    np_img = Image.fromarray(np_array)
    np_img = np_img.convert("RGB")

    width, height = np_img.size

    num_levels = 18
    image_pyramid_dict = {}
    current_img = np_img

    for i in tqdm(range(num_levels + 1), desc="Creating image pyramid"):
        level = num_levels - i

        current_img = current_img.resize(
            (max(1, int(width // 2**i)), max(1, int(height // 2**i)))
        )
        image_pyramid_dict[level] = current_img

    coordinates_level_pairs = get_tile_coordinate_level_pairs(
        image_pyramid_dict, tile_size=tile_size
    )

    # randomly generate a number between 0 and 1000000
    random_num = random.randint(0, 1000000)

    # the h5_tmp_path is in the save directory as the h5_path, but with tmp_random_num as name
    h5_tmp_path = os.path.join(os.path.dirname(h5_path), f"tmp_{random_num}.h5")
    save_pyramid_to_h5(image_pyramid_dict, h5_tmp_path)

    print(f"Saving temporary image pyramid to {h5_tmp_path}")
    print(
        f"Size (GiB) of the temporary h5 file: {os.path.getsize(h5_tmp_path) / 1e9} GiB"
    )

    # Initialize the final HDF5 file
    initialize_final_h5py_file(
        h5_path, width, height, num_levels=num_levels, patch_size=tile_size
    )

    list_of_batches = create_list_of_batches_from_list(
        coordinates_level_pairs, region_cropping_batch_size
    )

    ray.shutdown()
    ray.init(ignore_reinit_error=True)

    task_managers = [H5PyramidCropManager.remote(h5_tmp_path) for _ in range(num_cpus)]

    for manager in task_managers:
        manager.open_h5_file.remote()

    tasks = {}

    for i, batch in enumerate(list_of_batches):
        manager = task_managers[i % num_cpus]
        task = manager.async_get_jpeg_string_batch.remote(batch)
        tasks[task] = batch

    with h5py.File(h5_path, "a") as f:
        with tqdm(
            total=len(coordinates_level_pairs), desc="Cropping heatmap regions"
        ) as pbar:
            while tasks:
                done_ids, _ = ray.wait(list(tasks.keys()))

                for done_id in done_ids:
                    try:
                        batch = ray.get(done_id)
                        for indices_jpeg in batch:
                            x, y, dz_level, jpeg_string = indices_jpeg
                            f[str(dz_level)][x, y] = jpeg_string

                        pbar.update(len(batch))

                    except ray.exceptions.RayTaskError as e:
                        print(f"Task for batch {tasks[done_id]} failed with error: {e}")

                    del tasks[done_id]

    ray.shutdown()


if __name__ == "__main__":
    slide_np_path = "/media/hdd3/neo/viewer_sample_huong/website/390359_mask.npy"
    save_dir = "/media/hdd3/neo/viewer_sample_huong/website/test_tmp_dir"
    h5_path = os.path.join(save_dir, "test_np_dzsave.h5")

    # if the save_dir already exists, delete it
    if os.path.exists(save_dir):
        os.system(f"rm -r {save_dir}")

    os.makedirs(save_dir, exist_ok=True)

    # # Load the numpy arrays
    # start_time = time.time()
    # slide_np = np.load(slide_np_path)
    # print(f"Time taken to load numpy array: {time.time() - start_time} seconds")

    # # convert the numpy array to an image
    # start_time = time.time()
    # slide_img = Image.fromarray(slide_np)
    # # make sure the image is in RGB mode
    # slide_img = slide_img.convert("RGB")
    # print(
    #     f"Time taken to convert numpy array to image: {time.time() - start_time} seconds"
    # )

    # height, width = slide_img.size

    # # create an image pyramid with 18 levels
    # start_time = time.time()
    # num_levels = 18
    # image_pyramid_dict = {}
    # current_img = slide_img
    # for i in tqdm(range(num_levels + 1), desc="Creating image pyramid"):
    #     level = num_levels - i

    #     current_img = current_img.resize(
    #         (max(1, int(width // 2**i)), max(1, int(height // 2**i)))
    #     )
    #     image_pyramid_dict[level] = current_img
    # print(f"Time taken to create image pyramid: {time.time() - start_time} seconds")

    # # save the image pyramid to an HDF5 file
    # start_time = time.time()
    # h5_path_tmp = os.path.join(save_dir, "tmp.h5")
    # save_pyramid_to_h5(image_pyramid_dict, h5_path_tmp)
    # print(
    #     f"Time taken to save image pyramid to HDF5: {time.time() - start_time} seconds"
    # )

    # # now test the file size of the h5 file
    # h5_file_size = os.path.getsize(h5_path_tmp)

    # print(f"Size of the temporary h5 file: {h5_file_size} bytes")

    start_time = time.time()
    dzsave_h5_np(
        slide_np_path,
        h5_path=h5_path,
        tile_size=256,
        num_cpus=32,
        region_cropping_batch_size=256,
    )

    print(
        f"H5 file and heatmap created successfully. Time taken: {time.time() - start_time} seconds."
    )

    print(f"Size of the final h5 file: {os.path.getsize(h5_path) / 1e9} GiB")
