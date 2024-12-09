import os
import pandas as pd
from dzsave_neo import dzsave_neo
from tqdm import tqdm

dir_path = (
    "/media/ssd2/huong/meningioma_train_log/pancreas_new/uni/predict/attn_mil/heatmap"
)

# get all the files in the directory
files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]

splitter1 = "_rainbow_heatmap_mask"
splitter2 = "_coolwarm_heatmap_mask"

processed_files = []

for file in files:
    file = file.split(splitter1)[0]
    file = file.split(splitter2)[0]

    processed_files.append(file)

print(processed_files)

svs_dir = "/media/ssd1/huong/dataset_cyto_pancreas_203_imgs/diagnosis_203B"

# get the list of all the svs files in the svs directory
svs_files = [f for f in os.listdir(svs_dir) if f.endswith(".svs")]

svs_files_to_keep = []

for svs_file in tqdm(svs_files, desc="Checking SVS Files"):
    to_keep = False

    for processed_file in processed_files:
        if processed_file in svs_file:
            to_keep = True
            break

    if to_keep:
        svs_files_to_keep.append(svs_file)

save_dir = "/media/ssd2/neo/cp_aws_playground/dzsave_dir"
metadata_dict = {
    "svs_path": [],
    "h5_path": [],
    "tiling_time": [],
    "svs_file_size_mb": [],
    "h5_file_size_mb": [],
}

print(f"Found a total of {len(svs_files_to_keep)} SVS files.")

# get the list of all the svs files in the svs directory
svs_files_to_keep = svs_files_to_keep[:1]

# loop through each svs file
for svs_file in tqdm(svs_files_to_keep, desc="Tiling SVS Files"):
    # get the path to the svs file
    svs_path = os.path.join(svs_dir, svs_file)

    # get the path to the h5 file
    h5_file = svs_file.replace(".svs", ".h5")
    h5_path = os.path.join(save_dir, h5_file)

    # get the file size of the svs file
    svs_file_size_mb = os.path.getsize(svs_path) / 1024 / 1024

    # tiling the svs file
    tiling_time = dzsave_neo(svs_path, h5_path)

    # get the file size of the h5 file
    h5_file_size_mb = os.path.getsize(h5_path) / 1024 / 1024

    # add the metadata to the dictionary
    metadata_dict["svs_path"].append(svs_path)
    metadata_dict["h5_path"].append(h5_path)
    metadata_dict["tiling_time"].append(tiling_time)
    metadata_dict["svs_file_size_mb"].append(svs_file_size_mb)
    metadata_dict["h5_file_size_mb"].append(h5_file_size_mb)

    print(f"Finished tiling {svs_file} to {h5_file}", flush=True)
    print(f"Time taken: {tiling_time:.2f} seconds", flush=True)
    print(f"SVS file size: {svs_file_size_mb:.2f} MB", flush=True)
    print(f"H5 file size: {h5_file_size_mb:.2f} MB", flush=True)


# create a DataFrame from the metadata dictionary
metadata_df = pd.DataFrame(metadata_dict)

# save the metadata DataFrame to a CSV file
metadata_csv_path = os.path.join(save_dir, "tiling_metadata.csv")

metadata_df.to_csv(metadata_csv_path, index=False)
