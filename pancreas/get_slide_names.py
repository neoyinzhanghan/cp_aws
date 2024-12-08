import os
from tqdm import tqdm

dir_path = "/media/ssd2/huong/meningioma_train_log/pancreas_new/uni/predict/attn_mil/heatmap"

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

print(f"Found a total of {len(svs_files_to_keep)} SVS files.")  
