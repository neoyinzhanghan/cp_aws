import pandas as pd

old_df_path = (
    "/home/dog/Documents/neo/cp-lab-wsi-upload/wsi-and-heatmaps/pancreas_metadata.csv"
)

new_df_path = "/home/dog/Documents/huong/analysis/visualization/website/pancreas/pancreas_process_list.csv"

# open the df_path as a pandas DataFrame
df = pd.read_csv(old_df_path)
new_df = pd.read_csv(new_df_path)

# iterate through each row in the df
for idx, row in df.iterrows():
    case_name = row["case_name"]

    new_df_row = new_df[new_df["case_name"] == case_name]

    if new_df_row.empty:
        raise ValueError(f"No matching file found for {case_name}")

    prediction = new_df_row["pred"].values[0]

    print(prediction)