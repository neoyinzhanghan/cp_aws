# import pandas

# original_csv_path = "/Users/neo/Documents/MODS/cp_aws/pancreas/filter_prototype/original_pancreas_metadata.csv"
# csv_save_path = (
#     "/Users/neo/Documents/MODS/cp_aws/pancreas/filter_prototype/pancreas_metadata.csv"
# )

# # first open the original csv file
# df = pandas.read_csv(original_csv_path)

# # add a column named "project" with values "pancreas"
# df["project"] = "pancreas"

# # now create a new column named "group" with values matching the column "label"
# df["group"] = df["label"]

# # within each group, the new column "within_group_sorting" should be an integer ranking based on the "benign_prob" column
# df["within_group_sorting"] = (
#     df.groupby("group")["benign_prob"].rank(method="dense", ascending=True).astype(int)
# )

# # now create a new column named "display_name" with values matching the column "case_name"
# df["display_name"] = df["case_name"]

# # save the new DataFrame to a new csv file
# df.to_csv(csv_save_path, index=False)

import pandas

original_csv_path = "/Users/neo/Documents/MODS/cp_aws/pancreas/filter_prototype/original_pancreas_metadata.csv"
csv_file_with_grouping = "/Users/neo/Documents/MODS/cp_aws/pancreas/filter_prototype/pancreas_process_list.csv"
save_path = (
    "/Users/neo/Documents/MODS/cp_aws/pancreas/filter_prototype/pancreas_metadata.csv"
)

# first open the original csv file
original_df = pandas.read_csv(original_csv_path)

# add a column named "project" with values "pancreas"
original_df["project"] = "pancreas"

# now create a new column named "display_name" with values matching the column "case_name"
original_df["display_name"] = original_df["case_name"]

# open the csv file with grouping
grouping_df = pandas.read_csv(csv_file_with_grouping)

# iterate through each row in the original_df
for idx, row in original_df.iterrows():
    case_name = row["case_name"]

    grouping_df_row = grouping_df[grouping_df["case_name"] == case_name]

    if grouping_df_row.empty:
        raise ValueError(f"No matching file found for {case_name}")

    group = grouping_df_row["group"].values[0]
    within_group_sorting = grouping_df_row["group_order"].values[0]

    original_df.loc[idx, "group"] = group
    original_df.loc[idx, "group_order"] = within_group_sorting

# save the new DataFrame to a new csv file
original_df.to_csv(save_path, index=False)
