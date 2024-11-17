import os 
import torch
import openslide    
from torch.utils.data import Dataset, DataLoader

class LowMagRegionDataset(Dataset):
    """
    === Attributes ===
    slide: the openslide object of the slide
    level_3_coords: the coordinates of all the level 3 regions
    tile_size: the size of the tiles
    tile_size_level_3: the size of the tiles at level 3
    """

    def __init__(self, slide, tile_size=512):
        self.tile_size = tile_size
        self.tile_size_level_3 = tile_size // 8
        self.slide = slide

        # Get the dimensions of the slide at level 0
        self.slide_width = self.slide.dimensions[0]
        self.slide_height = self.slide.dimensions[1]

        # Get the coordinates of all the level 3 regions
        self.level_0_coords = self.get_level_0_coords()

    def get_level_0_coords(self):
        """
        Get the coordinates of all the level 3 regions
        """
        level_0_coords = []
        for x in range(self.slide_width // self.tile_size):
            for y in range(self.slide_height // self.tile_size):
                level_0_coords.append((x, y))

        return level_0_coords
    
    def __len__(self):
        return len(self.level_0_coords)
    
    def __getitem__(self, idx):
        x, y = self.level_0_coords[idx]
        region = self.slide.read_region(location=(x * self.tile_size, y * self.tile_size), level=3, size=(self.tile_size_level_3, self.tile_size_level_3))
        region = region.convert("RGB")
        # region = torch.tensor(region).permute(2, 0, 1).float() / 255.0
        return region, (x, y)