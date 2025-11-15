import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler



def read_unprocessed_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)
    n = len(df)
    n_cols = len(df.columns) - 2 #two response variables
    new_col_names = ["y1", "y2"]
    for i in range(n_cols):
        new_col_names.append(f"x{i+1}")
    df.columns = new_col_names
    df.drop(columns=["x27", "x33"], axis=1, inplace=True)
    df["x3_0"] = np.where(df["x3"] == 0, 1, 0)
    df["x4_0"] = np.where(df["x4"] == 0, 1, 0)
    return df

def make_cv_splits(data: pd.DataFrame, k: int=8) -> None:
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    data['cv_fold'] = -1

    for fold_number, (train_idx, val_idx) in enumerate(kf.split(data)):
        data.loc[val_idx, 'cv_fold'] = fold_number
    

def get_folds(target_mode: int, scaler_type: str=None) -> dict:
    """ target_mode
                    1: y1 (binary)
                    2: y2 (regression)
                    3: [y1, y2] (simultanous)
        scaler
                    minmax
                    standard
                    no scaling"""
    
    df = pd.read_csv("documents/data/processed_data.csv")

    match target_mode:
        case 1:
            target = "y1"
            cols_to_scale = ["x1", "x2", "x3", "x4", "x5"]
            cols_to_drop = ["y1", "cv_fold"]
            df = df.drop(columns=["y2"], axis=1)

        case 2:
            target = "y2"
            cols_to_scale = ["y2","x1", "x2", "x3", "x4", "x5"]
            cols_to_drop = ["y2", "cv_fold"]
            df = df.drop(columns=["y1"], axis=1)

        case _:
            cols_to_scale = ["y2","x1", "x2", "x3", "x4", "x5"]
            cols_to_drop = ["cv_fold"]
            target = ["y1", "y2"]

    match scaler_type:
        case "minmax":
            scaler = MinMaxScaler(clip=True)
        case "standard":
            scaler = StandardScaler()
        case _:
            scaler = None

    n_folds = len(pd.unique(df["cv_fold"]))
    folds = {}

    for fold in range(n_folds):

        inner_fold_numbers = [i for i in range(n_folds) if not i == fold]
        print(f"outer fold: {fold} training on: {inner_fold_numbers}")

        train_df = df[df["cv_fold"] != fold]
        holdout_df = df[df["cv_fold"] == fold]

        if not scaler is None:
            scaler.fit(train_df[cols_to_scale])
            train_df[cols_to_scale] = scaler.transform(train_df[cols_to_scale])
            holdout_df[cols_to_scale] = scaler.transform(holdout_df[cols_to_scale])

        train_X = train_df.drop(columns=cols_to_drop, axis=1).to_numpy()
        train_y = train_df[target].to_numpy()

        holdout_X = holdout_df.drop(columns=cols_to_drop, axis=1).to_numpy()
        holdout_y = holdout_df[target].to_numpy()

        inner_folds = []
        for inner_fold in inner_fold_numbers:
            if not inner_fold == fold:
                
                inner_test_df = train_df[train_df["cv_fold"] == inner_fold]
                inner_test_X = inner_test_df.drop(columns=cols_to_drop, axis=1).to_numpy()
                inner_test_y = inner_test_df[target].to_numpy()
                inner_train_df = train_df[train_df["cv_fold"] != inner_fold]
                inner_train_X = inner_train_df.drop(columns=cols_to_drop, axis=1).to_numpy()
                inner_train_y = inner_train_df[target].to_numpy()
                
                inner_folds.append({"train_X": inner_train_X, "train_y": inner_train_y, "test_X": inner_test_X, "test_y": inner_test_y})

        folds[fold] = {"holdout_X": holdout_X, "holdout_y": holdout_y, "train_X": train_X, "train_y": train_y, "inner_folds": inner_folds} 

    return folds



def main():
    path = "documents/data/GroupAssignment-Data.csv"
    df = read_unprocessed_data(path=path)
    make_cv_splits(df)
    file_name = "documents/data/processed_data.csv"
    df.to_csv(file_name, index=False)



if __name__ == "__main__":
    main()
