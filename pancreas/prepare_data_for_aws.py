import os
import pandas as pd
from tqdm import tqdm

wsi_h5_dir = "/media/ssd2/neo/cp_aws_playground/dzsave_dir"
metadata_csv_path = "/media/ssd2/neo/cp_aws_playground/pancreas_process_list.csv"

# get the list of all the h5 files in the wsi_h5 directory
wsi_h5_files = [
    f
    for f in os.listdir(wsi_h5_dir)
    if os.path.isfile(os.path.join(wsi_h5_dir, f)) and f.endswith(".h5")
]

final_metadata_dict = {
    "filename": [],
    "heatmap_filename": [],
    "pseudo_idx": [],
    "old_filename": [],
    "old_heatmap_filename": [],
    "case_name": [],
    "benign_prob": [],
    "low_grade_prob": [],
    "malignant_prob": [],
    "non_diagnosis_prob": [],
    "label": [],
    "split": [],
}

# open the metadata_csv_path
metadata_df = pd.read_csv(metadata_csv_path)


def find_wsi_h5_file(case_name):
    for wsi_h5_file in wsi_h5_files:
        if case_name in wsi_h5_file:
            return wsi_h5_file

    raise ValueError(f"No matching file found for {case_name}")


# iterate through each row in the metadata_df
for idx, row in tqdm(metadata_df.iterrows(), desc="Processing Metadata CSV"):
    case_name = row["case_name"]

    wsi_h5_file = find_wsi_h5_file(case_name)
