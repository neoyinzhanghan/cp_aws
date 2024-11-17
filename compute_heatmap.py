import h5py
import openslide
import numpy as np
from utils import smooth_function
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
from dataset import LowMagRegionDataset
from torch.utils.data import DataLoader
from BMARegionClfManager import load_clf_model, predict_batch
from BMAassumptions import region_clf_ckpt_path

batch_size = 256
num_workers = 32


def generate_red_green_heatmap(matrix):
    """
    Generates a heatmap where values closer to 0 are red and closer to 1 are green.

    Parameters:
    - matrix (numpy.ndarray): A 2D array with values between 0 and 1.

    Returns:
    - heatmap_pil (PIL.Image.Image): The heatmap image as a PIL Image object.
    """
    # Define the custom colormap: red for 0, green for 1
    red_green_cmap = LinearSegmentedColormap.from_list("RedGreen", ["red", "green"])

    # Normalize the matrix to range [0, 1] if needed (you mentioned values are between 0 and 1)
    normalized_matrix = np.clip(matrix, 0, 1)  # Ensure values are in [0, 1]

    # Apply the colormap to the matrix
    heatmap_image = red_green_cmap(normalized_matrix)

    # Remove the alpha channel from the resulting image (if it exists)
    heatmap_image = (heatmap_image[:, :, :3] * 255).astype(np.uint8)

    # Convert the NumPy array to a PIL Image
    heatmap_pil = Image.fromarray(heatmap_image)

    return heatmap_pil


# Custom collate function to handle PIL images and names
def custom_collate_fn(batch):
    # Batch is a list of tuples (PIL image, image_name)
    pil_images, coordinates = zip(*batch)
    return list(pil_images), list(coordinates)


def dyadic_average_downsample_heatmap(float_matrix):
    """
    Downsample the heatmap by averaging the values in 2x2 blocks.

    Parameters:
    - float_matrix (np.ndarray): The heatmap as a float numpy array.

    Returns:
    - np.ndarray: The downsampled heatmap.
    """
    # Get the dimensions of the input matrix
    height, width = float_matrix.shape

    # remove the last row and column if the height or width is odd
    if height % 2 != 0:
        float_matrix = float_matrix[:-1]
        height -= 1
    if width % 2 != 0:
        float_matrix = float_matrix[:, :-1]
        width -= 1

    # Compute the new dimensions after downsampling
    new_height = height // 2
    new_width = width // 2

    # Reshape the input matrix to a 4D tensor
    reshaped_matrix = float_matrix.reshape(new_height, 2, new_width, 2)

    # Average the values in each 2x2 block
    downsampled_matrix = reshaped_matrix.mean(axis=(1, 3))

    return downsampled_matrix


class HeatMapTileMaker:
    """
    === Attributes ===
    - slide_path: the path to the slide image
    - slide: the openslide object of
    - tile_size: the size of the tiles
    - dataset: the dataset object to extract tiles from
    - dataloader: the dataloader object to load the tiles
    - heatmap: the heatmap of the slide stored at the highest resolution, it is a float tensor where heatmap[x, y] is the confidence score of the region at (x, y)
    - model: the classifier model to predict the heatmap

    """

    def __init__(self, slide_path, tile_size=512):
        self.slide_path = slide_path
        self.tile_size = tile_size
        self.slide = openslide.OpenSlide(self.slide_path)
        self.dataset = LowMagRegionDataset(self.slide, self.tile_size)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=custom_collate_fn,
        )
        # Load the model
        self.model = load_clf_model(region_clf_ckpt_path)

        # shape of the heatmap should be slide_width_level_0 // 512, slide_height_level_0 // 512, we start by initializing it to zeros numpy array
        self.heatmap = np.zeros(
            (self.slide.dimensions[0] // 512, self.slide.dimensions[1] // 512)
        )
        self.dz_heatmap_dict = {}

    def compute_heatmap(self):

        largest_score = 0
        # Iterate through the dataset with a DataLoader and progress bar
        for pil_images, coordinates in tqdm(self.dataloader, desc="Processing Batches"):
            # Predict batch of images
            scores = predict_batch(pil_images, self.model)

            for i, (x, y) in enumerate(coordinates):
                # Update the heatmap with the confidence score, as a float
                self.heatmap[x, y] = scores[i]

                # Update the largest score if needed
                if scores[i] > largest_score:
                    largest_score = scores[i]

        self.dz_heatmap_dict[18] = self.heatmap

        current_heatmap = self.heatmap

        for level in range(18 - 1, -1, -1):
            current_heatmap = dyadic_average_downsample_heatmap(current_heatmap)
            self.dz_heatmap_dict[level] = current_heatmap

        print(f"Largest score: {largest_score}")

    def get_heatmap_values(self, level, x, y):
        """
        Get the heatmap values at a specific level and location.

        Parameters:
        - level (int): The level of the heatmap.
        - x (int): The x-coordinate of the location.
        - y (int): The y-coordinate of the location.

        Returns:
        - float: The heatmap value at the specified location.
        """

        try:
            return float(
                self.dz_heatmap_dict[level][x, y]
            )  # if index out of bounds, return 0
        except IndexError:
            return float(0)

    def get_gaussian_heatmap_values(self, x, y):
        """
        Get the heatmap values at a specific level and location using a gaussian kernel centered at the center of the slide, with sigma = 1/3 of the slide height.
        """
        slide_width, slide_height = self.slide.dimensions
        center_x = 2 * (slide_width // self.tile_size) // 3
        center_y = 1 * (slide_height // self.tile_size) // 3
        sigma = (slide_height // self.tile_size) / 3
        heatmap_values = np.exp(
            -((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * sigma**2)
        )
        return heatmap_values

    def get_heatmap_image(self, level, x, y):
        """
        Get the heatmap overlay for a specific level and location.

        Parameters:
        - level (int): The level of the heatmap.
        - x (int): The x-coordinate of the location.
        - y (int): The y-coordinate of the location.

        Returns:
        - np.ndarray: The heatmap overlay as a NumPy array.
        """

        openslide_level = 18 - level
        heatmap_grid_size = 512 // (2 ** (openslide_level))

        heatmap_overlay_score = np.zeros((512, 512))

        for i in range(2 ** (openslide_level)):
            for j in range(2 ** (openslide_level)):
                # heatmap_overlay_score[j*heatmap_grid_size:(j+1)*heatmap_grid_size, i*heatmap_grid_size:(i+1)*heatmap_grid_size] = self.get_gaussian_heatmap_values(x*(2**(openslide_level)) + i, y*(2**(openslide_level)) + j)
                heatmap_overlay_score[
                    j * heatmap_grid_size : (j + 1) * heatmap_grid_size,
                    i * heatmap_grid_size : (i + 1) * heatmap_grid_size,
                ] = self.get_heatmap_values(
                    18,
                    x * 2 ** (openslide_level) + i,
                    y * 2 ** (openslide_level) + j,
                )

        return generate_red_green_heatmap(heatmap_overlay_score)

    def save_heatmap_to_h5(self, heatmap_h5_save_path):
        # save the self.dz_heatmap_dict[18] to the h5 file with a key "heatmap"

        with h5py.File(heatmap_h5_save_path, "w") as f:
            f.create_dataset("heatmap", data=self.dz_heatmap_dict[18])

        print(f"Saved heatmap to {heatmap_h5_save_path}")


def create_heatmap_to_h5(slide_path, heatmap_h5_save_path):
    heatmap_tile_maker = HeatMapTileMaker(slide_path=slide_path, tile_size=512)
    heatmap_tile_maker.compute_heatmap()
    heatmap_tile_maker.save_heatmap_to_h5(heatmap_h5_save_path)


class HeatMapTileLoader:
    """ """

    def __init__(self, np_heatmap, tile_size=512):
        self.tile_size = tile_size
        # shape of the heatmap should be slide_width_level_0 // 512, slide_height_level_0 // 512, we start by initializing it to zeros numpy array
        self.heatmap = np_heatmap
        self.dz_heatmap_dict = {}

    def compute_heatmap(self):

        largest_score = 0
        self.dz_heatmap_dict[18] = self.heatmap

        current_heatmap = self.heatmap

        for level in range(18 - 1, -1, -1):
            current_heatmap = dyadic_average_downsample_heatmap(current_heatmap)
            self.dz_heatmap_dict[level] = current_heatmap

        print(f"Largest score: {largest_score}")

    def get_heatmap_values(self, level, x, y):
        """
        Get the heatmap values at a specific level and location.

        Parameters:
        - level (int): The level of the heatmap.
        - x (int): The x-coordinate of the location.
        - y (int): The y-coordinate of the location.

        Returns:
        - float: The heatmap value at the specified location.
        """

        try:
            return float(
                self.dz_heatmap_dict[level][x, y]
            )  # if index out of bounds, return 0
        except IndexError:
            return float(0)

    def get_heatmap_image(self, level, x, y):
        """
        Get the heatmap overlay for a specific level and location.

        Parameters:
        - level (int): The level of the heatmap.
        - x (int): The x-coordinate of the location.
        - y (int): The y-coordinate of the location.

        Returns:
        - np.ndarray: The heatmap overlay as a NumPy array.
        """

        openslide_level = 18 - level
        heatmap_grid_size = 512 // (2 ** (openslide_level))

        heatmap_overlay_score = np.zeros((512, 512))

        for i in range(2 ** (openslide_level)):
            for j in range(2 ** (openslide_level)):
                # heatmap_overlay_score[j*heatmap_grid_size:(j+1)*heatmap_grid_size, i*heatmap_grid_size:(i+1)*heatmap_grid_size] = self.get_gaussian_heatmap_values(x*(2**(openslide_level)) + i, y*(2**(openslide_level)) + j)
                heatmap_overlay_score[
                    j * heatmap_grid_size : (j + 1) * heatmap_grid_size,
                    i * heatmap_grid_size : (i + 1) * heatmap_grid_size,
                ] = self.get_heatmap_values(
                    18,
                    x * 2 ** (openslide_level) + i,
                    y * 2 ** (openslide_level) + j,
                )

        return generate_red_green_heatmap(heatmap_overlay_score)

    def save_heatmap_to_h5(self, heatmap_h5_save_path):
        # save the self.dz_heatmap_dict[18] to the h5 file with a key "heatmap"

        with h5py.File(heatmap_h5_save_path, "w") as f:
            f.create_dataset("heatmap", data=self.dz_heatmap_dict[18])

        print(f"Saved heatmap to {heatmap_h5_save_path}")


def get_heatmap_overlay(region, heatmap_image, alpha=0.5):
    heatmap_image = np.array(heatmap_image.convert("RGB"))
    if region.shape[2] != 3:
        raise ValueError("Region image must be in RGB format with 3 channels")
    region = region.astype(np.float32) / 255.0
    heatmap_image = heatmap_image.astype(np.float32) / 255.0
    overlay_image_np = (1 - alpha) * region + alpha * heatmap_image
    overlay_image_np = np.clip(overlay_image_np, 0, 1)
    overlay_image_np = (overlay_image_np * 255).astype(np.uint8)
    return Image.fromarray(overlay_image_np)


if __name__ == "__main__":
    slide_path = (
        "/media/hdd3/neo/tmp_slide_dir/H19-5749;S10;MSKI - 2023-05-24 21.38.53.ndpi"
    )
    heatmap_h5_save_path = "/media/hdd3/neo/S3_tmp_dir/bma_test_slide_heatmap.h5"
    create_heatmap_to_h5(slide_path, heatmap_h5_save_path)
