import ray
import h5py
import openslide

patch_grid_csv = (
    "/home/dog/Documents/huong/analysis/visualization/website/mayo/K106022_coords.csv"
)
wsi_path = "/media/ssd2/huong/mayo_bbd/test_visual/process_img_list/K106022.svs"


# open the slide and print the level 0 dimensions
slide = openslide.OpenSlide(wsi_path)
print(slide.level_dimensions[0])
