import os
import time
import pandas as pd
from tqdm import tqdm
from dzsave_npy_heatmap import dzsave_npy_heatmap

dzsave_dir = "/media/ssd2/neo/cp_aws_playground/dzsave_dir"

heatmap_dir = (
    "/media/ssd2/huong/meningioma_train_log/pancreas_new/uni/predict/attn_mil/heatmap"
)

heatmap_save_dir = "/media/ssd2/neo/cp_aws_playground/dzsave_dir_rainbow_heatmap"

os.makedirs(heatmap_save_dir, exist_ok=True)

# first list all the files in the dzsave_dir
dzsave_files = [
    f
    for f in os.listdir(dzsave_dir)
    if os.path.isfile(os.path.join(dzsave_dir, f)) and f.endswith(".h5")
]

dzsave_files = dzsave_files[:2]  # TODO remove this line once the code is working

metadata_dict = {
    "h5_path": [],
    "heatmap_npy_path": [],
    "heatmap_h5_path": [],
    "tiling_time": [],
    "h5_file_size_mb": [],
    "heatmap_npy_file_size_mb": [],
    "heatmap_h5_file_size_mb": [],
}

# the h5 file should have file_name.h5, and the npy file should have file_name_rainbow_heatmap_mask.npy
for dzsave_file in tqdm(dzsave_files, desc="dzsaving Heatmap NPY Files"):

    start_time = time.time()
    heatmap_file_name = dzsave_file.split(".h5")[0] + "_rainbow_heatmap_mask.npy"
    heatmap_file_path = os.path.join(heatmap_dir, heatmap_file_name)

    dzsave_file_path = os.path.join(dzsave_dir, dzsave_file)

    heatmap_h5_save_path = os.path.join(
        heatmap_save_dir, dzsave_file.split(".h5")[0] + "_rainbow_heatmap.h5"
    )

    dzsave_npy_heatmap(dzsave_file_path, heatmap_h5_save_path, heatmap_file_path)

    metadata_dict["h5_path"].append(dzsave_file_path)
    metadata_dict["heatmap_npy_path"].append(heatmap_file_path)
    metadata_dict["heatmap_h5_path"].append(heatmap_h5_save_path)
    metadata_dict["tiling_time"].append(time.time() - start_time)
    metadata_dict["h5_file_size_mb"].append(
        os.path.getsize(dzsave_file_path) / (1024 * 1024)
    )
    metadata_dict["heatmap_npy_file_size_mb"].append(
        os.path.getsize(heatmap_file_path) / (1024 * 1024)
    )
    metadata_dict["heatmap_h5_file_size_mb"].append(
        os.path.getsize(heatmap_h5_save_path) / (1024 * 1024)
    )

metadata_df = pd.DataFrame(metadata_dict)
metadata_df.to_csv(
    os.path.join(heatmap_save_dir, "dzsave_rainbow_heatmap_metadata.csv"), index=False
)
