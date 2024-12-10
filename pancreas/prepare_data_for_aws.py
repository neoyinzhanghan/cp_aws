import os
import pandas as pd
import shutil
from tqdm import tqdm

wsi_h5_dir = "/media/ssd2/neo/cp_aws_playground/dzsave_dir"
heatmap_h5_dir = "/media/ssd2/neo/cp_aws_playground/dzsave_dir_rainbow_heatmap"
metadata_csv_path = "/media/ssd2/neo/cp_aws_playground/pancreas_process_list.csv"

wsi_h5_save_dir = "/home/dog/Documents/neo/cp-lab-wsi-upload/wsi-and-heatmaps"
heatmap_h5_save_dir = (
    "/home/dog/Documents/neo/cp-lab-wsi-upload/wsi-and-heatmaps/heatmaps"
)

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
for idx, row in tqdm(
    metadata_df.iterrows(), desc="Processing Metadata CSV", total=len(metadata_df)
):
    case_name = row["case_name"]

    wsi_h5_file = find_wsi_h5_file(case_name)
    heatmap_h5_file = wsi_h5_file.split(".h5")[0] + "_rainbow_heatmap.h5"

    old_filename = wsi_h5_file
    old_heatmap_filename = heatmap_h5_file

    new_filename = str(idx + 1) + ".h5"
    new_heatmap_filename = str(idx + 1) + "_heatmap.h5"

    new_filepath = os.path.join(wsi_h5_save_dir, new_filename)
    new_heatmap_filepath = os.path.join(heatmap_h5_save_dir, new_heatmap_filename)

    benign_prob = row["benign"]
    low_grade_prob = row["low-grade"]
    malignant_prob = row["malignant"]
    non_diagnosis_prob = row["non-diagnosis"]
    label = row["label"]
    split = row["split"]

    # copy the wsi_h5 file to the wsi_h5_save_dir # TODO Temporarily commented out
    shutil.copyfile(os.path.join(wsi_h5_dir, wsi_h5_file), new_filepath)
    shutil.copyfile(os.path.join(heatmap_h5_dir, heatmap_h5_file), new_heatmap_filepath)

    final_metadata_dict["filename"].append(new_filename)
    final_metadata_dict["heatmap_filename"].append(new_heatmap_filename)
    final_metadata_dict["pseudo_idx"].append(idx + 1)
    final_metadata_dict["old_filename"].append(old_filename)
    final_metadata_dict["old_heatmap_filename"].append(old_heatmap_filename)
    final_metadata_dict["case_name"].append(case_name)
    final_metadata_dict["benign_prob"].append(benign_prob)
    final_metadata_dict["low_grade_prob"].append(low_grade_prob)
    final_metadata_dict["malignant_prob"].append(malignant_prob)
    final_metadata_dict["non_diagnosis_prob"].append(non_diagnosis_prob)
    final_metadata_dict["label"].append(label)
    final_metadata_dict["split"].append(split)

# save the final metadata to a csv file
final_metadata_df = pd.DataFrame(final_metadata_dict)
final_metadata_df.to_csv(
    os.path.join(wsi_h5_save_dir, "pancreas_metadata.csv"), index=False
)
