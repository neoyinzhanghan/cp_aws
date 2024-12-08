import os

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