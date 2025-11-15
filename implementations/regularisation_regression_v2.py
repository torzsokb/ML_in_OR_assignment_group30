import pandas as pd
import numpy as np
import optuna

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

df = pd.read_csv("documents/data/processed_data.csv")
df = df.drop(columns=["y1"], axis=1)

n_folds = len(pd.unique(df["cv_fold"]))
folds = {}

for fold in range(n_folds):
    
    fold_numbers = list(range(n_folds))
    fold_numbers.remove(fold)
    outer_validation = fold + 1 if fold + 1 < n_folds else 0
    
    print(f"fold: {fold}, outer_validation: {outer_validation}")
    
    holdout_df = df[df["cv_fold"] == fold]
    holdout_data = {
        "X": holdout_df.drop(["y2", "cv_fold"], axis=1).values,
        "y": holdout_df["y2"].values
    }

    train_df = df[df["cv_fold"] != fold]
    outer_train_df_split = train_df[train_df["cv_fold"] != outer_validation]
    outer_train_data = {
        "X": outer_train_df_split.drop(["y2", "cv_fold"], axis=1).values,
        "y": outer_train_df_split["y2"].values
    }
    
    outer_validation_df = train_df[train_df["cv_fold"] == outer_validation]
    outer_validation_data = {
        "X": outer_validation_df.drop(["y2", "cv_fold"], axis=1).values,
        "y": outer_validation_df["y2"].values
    }

    inner_folds = []
    for i, inner_fold in enumerate(fold_numbers):

        print(f"\tinner fold: {inner_fold}")

        inner_test_df = train_df[train_df["cv_fold"] == inner_fold]
        inner_train_df = train_df[train_df["cv_fold"] != inner_fold]

        validation = fold_numbers[0] if i + 1 == len(fold_numbers) else fold_numbers[i+1]

        inner_train_df_split = inner_train_df[inner_train_df["cv_fold"] != validation]
        inner_validation_df = inner_train_df[inner_train_df["cv_fold"] == validation]

        inner_folds.append({
            "train": {
                "X": inner_train_df_split.drop(["y2", "cv_fold"], axis=1).values,
                "y": inner_train_df_split["y2"].values
            },
            "test": {
                "X": inner_test_df.drop(["y2", "cv_fold"], axis=1).values,
                "y": inner_test_df["y2"].values
            },
            "validation": {
                "X": inner_validation_df.drop(["y2", "cv_fold"], axis=1).values,
                "y": inner_validation_df["y2"].values
            }
        })

    folds[fold] = {
        "holdout": holdout_data,
        "train": outer_train_data,
        "validation": outer_validation_data,
        "inner_folds": inner_folds
    }

def inner_cv(outer_fold: int, params: dict):

    scores = []

    for inner_fold in folds[outer_fold]["inner_folds"]:
        
        model = make_pipeline(
            MinMaxScaler(),
            ElasticNet(
                alpha=params["alpha"],
                l1_ratio=params["l1_ratio"],
                max_iter=3000
            )
        )

        model.fit(inner_fold["train"]["X"], inner_fold["train"]["y"])

    
        pred = model.predict(inner_fold["test"]["X"])
        mse = mean_squared_error(inner_fold["test"]["y"], pred)  
        # rmse = np.sqrt(mse)                                      
        scores.append(mse)

    return np.mean(scores)

k = 0 

def objective(trial):

    params = {
        "alpha": trial.suggest_float("alpha", 1e-5, 10, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 0, 1),
    }

    return inner_cv(outer_fold=k, params=params)

def main():
    
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner()
    )
    
    study.optimize(objective, n_trials=100)

    print("Best parameters:", study.best_params)

    best_params = study.best_params

    final_model = make_pipeline(
        MinMaxScaler(),
        ElasticNet(
            alpha=best_params["alpha"],
            l1_ratio=best_params["l1_ratio"],
            max_iter=3000
        )
    )

    final_model.fit(folds[k]["train"]["X"], folds[k]["train"]["y"])

    pred = final_model.predict(folds[k]["holdout"]["X"])
    mse_holdout = mean_squared_error(folds[k]["holdout"]["y"], pred)
    # holdout_rmse = np.sqrt(mse_holdout)

    print(f"\nFinal Holdout RMSE (outer fold {k}): {mse_holdout:.4f}")


if __name__ == "__main__":
    main()
