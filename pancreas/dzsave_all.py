import os
import pandas as pd
from dzsave_final import dzsave
from tqdm import tqdm

svs_dir = "/media/ssd1/huong/dataset_cyto_pancreas_203_imgs/diagnosis_203B"
# save_dir = ""

metadata_dict = {
    "svs_path": [],
    "h5_path": [],
    "tiling_time": [],
    "svs_file_size_mb": [],
    "h5_file_size_mb": [],
}

# get the list of all the svs files in the svs directory
svs_files = [f for f in os.listdir(svs_dir) if f.endswith(".svs")]

print(f"Found a total of {len(svs_files)} SVS files.")

# # loop through each svs file
# for svs_file in tqdm(svs_files, desc="Tiling SVS Files"):
#     # get the path to the svs file
#     svs_path = os.path.join(svs_dir, svs_file)
    
#     # get the path to the h5 file
#     h5_file = svs_file.replace(".svs", ".h5")
#     h5_path = os.path.join(save_dir, h5_file)
    
#     # get the file size of the svs file
#     svs_file_size_mb = os.path.getsize(svs_path) / 1024 / 1024
    
#     # tiling the svs file
#     tiling_time = dzsave(svs_path, h5_path)
    
#     # get the file size of the h5 file
#     h5_file_size_mb = os.path.getsize(h5_path) / 1024 / 1024
    
#     # add the metadata to the dictionary
#     metadata_dict["svs_path"].append(svs_path)
#     metadata_dict["h5_path"].append(h5_path)
#     metadata_dict["tiling_time"].append(tiling_time)
#     metadata_dict["svs_file_size_mb"].append(svs_file_size_mb)
#     metadata_dict["h5_file_size_mb"].append(h5_file_size_mb)