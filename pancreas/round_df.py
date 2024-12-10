import pandas as pd
import numpy as np

df_path = (
    "/home/dog/Documents/neo/cp-lab-wsi-upload/wsi-and-heatmaps/pancreas_metadata.csv"
)

# open the df_path as a pandas DataFrame
df = pd.read_csv(df_path)

# apply rounding to 3 significant figures for every float entry in the DataFrame
df = df.applymap(lambda x: round(x, 3) if isinstance(x, (float, np.float64)) else x)

# save the rounded DataFrame to a new csv file
df.to_csv(df_path.replace(".csv", "_rounded.csv"), index=False)
