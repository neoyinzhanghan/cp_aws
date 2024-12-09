import os

dzsave_dir = "/media/ssd2/neo/cp_aws_playground/dzsave_dir"

heatmap_dir = (
    "/media/ssd2/huong/meningioma_train_log/pancreas_new/uni/predict/attn_mil/heatmap"
)

found_heatmap_npy = 0

# first list all the files in the dzsave_dir
dzsave_files = [
    f for f in os.listdir(dzsave_dir) if os.path.isfile(os.path.join(dzsave_dir, f)) and f.endswith(".h5")
]

# the h5 file should have file_name.h5, and the npy file should have file_name_rainbow_heatmap_mask.npy
for dzsave_file in dzsave_files:
    heatmap_file_name = dzsave_file.split(".h5")[0] + "_rainbow_heatmap_mask.npy"
    heatmap_file_path = os.path.join(heatmap_dir, heatmap_file_name)

    if os.path.exists(heatmap_file_path):
        found_heatmap_npy += 1

print(f"Total of {len(dzsave_files)} h5 files.")
print(f"Found {found_heatmap_npy} heatmap npy files.")
