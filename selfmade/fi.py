# from selfmade.genome import Population, Evaluator
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance

def find_data(root):
    i = 1
    dfs = []
    try:
        root = Path(root)
    except Exception:
        print("Error occurred.")
        raise

    while (root / f"exp{i:03d}").exists():
        df = pd.read_csv(root / f"exp{i:03d}" / "population.csv")
        dfs.append(df)
        i += 1
    return dfs

def harvest_data(stripped_cols = ['generation', 'test_accuracy', 'training_time', 
               'train_accuracy', 'parameters'], root = "records", data_path = None):

    if data_path:
        df_merged = pd.read_csv(data_path)
    else:
        dfs = find_data(Path(root))
        df_merged = pd.concat(dfs, ignore_index=True)

    unused_cols = stripped_cols

    df_new = df_merged.copy()
    df_new = df_new.drop(unused_cols, axis=1)
    
    return df_new

# for col in df_new.columns:
#     if col.startswith("layer"):
#         print(df_new[col].value_counts())

# print(df_new.dtypes)

# Code Expand + Encode

def expand_code(code):
    bits = []
    code = str(code)
    for gene in code.split("-"):
        bits.extend(int(bit) for bit in gene)
    return bits

def directions(dataframe : pd.DataFrame):
    res = []

    feature_cols = dataframe.drop(columns=["validation_accuracy"]).columns
    
    for col in feature_cols:
        group_1 = dataframe.loc[dataframe[col] == 1, "validation_accuracy"]
        group_0 = dataframe.loc[dataframe[col] == 0, "validation_accuracy"]

        mean_1 = group_1.mean()
        mean_0 = group_0.mean()

        std_1 = group_1.std()
        std_0 = group_0.std()

        direction = mean_1 - mean_0

        pooled_std = np.sqrt((std_1**2 + std_0**2) / 2)

        if pooled_std == 0 or np.isnan(pooled_std):
            strength = 0
        else:
            strength = abs(direction) / pooled_std

        res.append({"feature" : col, "mean_1": mean_1, "mean_0": mean_0, "direction": direction,
                    "std_1" : std_1, "std_0" : std_0, "direction_strength" : strength})
    direction_df = pd.DataFrame(res)
    return direction_df

def layer_tendency(dataframe : pd.DataFrame):
    layer_tendency = {}
    
    temp_df = dataframe.drop(columns=['validation_accuracy'])
    layer_cols = temp_df.columns.tolist()
    
    good = dataframe[dataframe["validation_accuracy"] >= dataframe["validation_accuracy"].quantile(0.8)]
    for layer in layer_cols:
        all_dist = dataframe[layer].value_counts(normalize=True)
        good_dist = good[layer].value_counts(normalize=True)
    
        tendency = good_dist.subtract(all_dist, fill_value=0)
        layer_tendency[layer] = tendency
    return layer_tendency

def preprocessing(df):
    
    bits = len(expand_code(df.loc[0, "code"]))
    code_cols = [f"bit{i+1}" for i in range(bits)]
    df_code = pd.DataFrame(df['code'].apply(expand_code).tolist(), columns=code_cols)

    df = df.drop(columns=["code"])
    df = pd.concat([df_code, df], axis=1)

    dir_df = pd.concat([df_code, df["validation_accuracy"]], axis=1)

    bit_dir = directions(dir_df)
    print(bit_dir)

    # Layer Encoding

    encoded = {}
    df_layers = df["validation_accuracy"]

    for col in df.columns:
        if col.startswith("layer"):
            le = LabelEncoder()
            df_layers = pd.concat([df_layers, df[col]], axis=1)        
            df[col] = le.fit_transform(df[col])
            encoded[col] = le
            
    layer_report = layer_tendency(df_layers)

    print(layer_report)
    return df, bit_dir, layer_report

def feature_importance(dataframe, device = "cpu"):
    
    if device == "gpu":
        import cudf
        from cuml.ensemble import RandomForestRegressor
        from cuml.model_selection import train_test_split
        from cuml.metrics import r2_score, mean_absolute_error
        dataframe = cudf.from_pandas(dataframe)
        rf = RandomForestRegressor(n_estimators=300, random_state=42, n_streams=4)
       
    elif device == "cpu":
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score, mean_absolute_error
        rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=1)

    else:
        print("Unknown device")
        return None, None, None
        
    y = dataframe["validation_accuracy"]
    X = dataframe.drop("validation_accuracy", axis=1)

    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    rf.fit(X_train, y_train)

    pred = rf.predict(X_test)

    print("R² :", r2_score(y_test, pred))
    print("MAE:", mean_absolute_error(y_test, pred))
            
    importance = rf.feature_importances_

    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": importance
    })

    # importance_df = importance_df.sort_values(
    #     "importance", ascending=False)

    # perm = permutation_importance(
    #     rf, X, y, n_repeats=10,
    #     random_state=42, n_jobs=1)

    perm = permutation_importance(
        rf, X_test, y_test, n_repeats=10,
        random_state=42, n_jobs=1)
    
    return importance, importance_df, perm

def pipeline(device = "cpu", data_path = None):
    df_full = harvest_data(data_path = data_path)
    df_processed, bit_dir, layer_report = preprocessing(df_full)
    imp, df_imp, perm = feature_importance(df_processed, device)
    
    from scipy.stats import spearmanr
    
    rf_imp = imp
    perm_imp = perm.importances_mean
    
    corr, p_val = spearmanr(rf_imp, perm_imp)

    output_dict = {"corr" : corr, "p_value" : p_val, "df_importance" : df_imp,
                    "bit_directions" : bit_dir, "layer_report" : layer_report
                   }
    
    return output_dict

# Testing ground

if __name__ == "__main__":
    # data_path = "/kaggle/input/datasets/an0rm4lp3rs0n/fi-dts/population.csv"
    data_path = "/kaggle/input/datasets/an0rm4lp3rs0n/fi-full/population_full.csv"
    results = pipeline(device = "gpu", data_path=data_path)
    # CPU fallback
    # results = pipeline()
    
    print("correlation: ", results["corr"])
    print("p-value: ", results["p_value"])
    print()
    df_imp = results["df_importance"]
    print(df_imp)

    df_imp.to_csv("fi.csv")