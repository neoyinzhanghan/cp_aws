import os
import openslide

svs_dir = "/media/ssd1/huong/dataset_cyto_pancreas_203_imgs/diagnosis_203B"

# get the path to all the svs files in the directory
svs_paths = [
    os.path.join(svs_dir, f) for f in os.listdir(svs_dir) if f.endswith(".svs")
]

print(f"Found {len(svs_paths)} svs files")

for svs_path in svs_paths:
    svs = openslide.OpenSlide(svs_path)

    try:
        # print the level 0 mpp for the current svs file
        print(
            f"Level 0 MPP for {svs_path}: {svs.properties[openslide.PROPERTY_NAME_MPP_X]}"
        )

    except Exception as e:
        print(f"Error reading mpp for {svs_path}: {e}")
