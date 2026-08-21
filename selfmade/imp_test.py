from selfmade.fi import find_data
import pandas as pd

df_full = pd.concat(find_data("records"), ignore_index=True)

df_full.to_csv("population_full.csv", index = False)