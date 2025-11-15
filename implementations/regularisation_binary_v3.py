import pandas as pd
import numpy as np
import optuna

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import KFold

df = pd.read_csv("documents/data/processed_data.csv")
df = df.drop(columns=["y2"])

X = df.drop(columns=["y1", "cv_fold"]).values
y = df["y1"].values

n_outer_folds = len(df["cv_fold"].unique())  
n_inner_folds = n_outer_folds - 1             
outer_folds = sorted(df["cv_fold"].unique())

folds = {}

for fold in outer_folds:

    holdout_mask = df["cv_fold"] == fold
    train_mask = df["cv_fold"] != fold

    X_holdout = X[holdout_mask]
    y_holdout = y[holdout_mask]

    X_outer_train = X[train_mask]
    y_outer_train = y[train_mask]

    inner_kf = KFold(n_splits=n_inner_folds, shuffle=True, random_state=42)
    inner_folds = []

    for train_idx, val_idx in inner_kf.split(X_outer_train):
        X_train_inner = X_outer_train[train_idx]
        y_train_inner = y_outer_train[train_idx]

        X_val_inner = X_outer_train[val_idx]
        y_val_inner = y_outer_train[val_idx]

        inner_folds.append({
            "train": {"X": X_train_inner, "y": y_train_inner},
            "validation": {"X": X_val_inner, "y": y_val_inner}
        })

    folds[fold] = {
        "outer_train": {"X": X_outer_train, "y": y_outer_train},
        "holdout": {"X": X_holdout, "y": y_holdout},
        "inner_folds": inner_folds
    }

def inner_cv(outer_fold, params):

    losses = []

    for inner_split in folds[outer_fold]["inner_folds"]:

        model = make_pipeline(
            MinMaxScaler(),
            LogisticRegression(
                C=params["C"],
                penalty=params["penalty"],
                solver=params["solver"],
                max_iter=5000
            )
        )

        model.fit(inner_split["train"]["X"], inner_split["train"]["y"])

        prob = model.predict_proba(inner_split["validation"]["X"])
        loss = log_loss(inner_split["validation"]["y"], prob)

        losses.append(loss)

    return np.mean(losses)

def objective(trial):

    params = {
        "C": trial.suggest_float("C", 1e-4, 100, log=True),
        "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
        "solver": trial.suggest_categorical("solver", ["liblinear", "saga"])
    }

    return inner_cv(outer_fold=0, params=params)  


def main():
    outer_fold = 0  
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=75)

    print("Best params:", study.best_params)

    best_params = study.best_params

    final_model = make_pipeline(
        MinMaxScaler(),
        LogisticRegression(
            C=best_params["C"],
            penalty=best_params["penalty"],
            solver=best_params["solver"],
            max_iter=5000
        )
    )

    final_model.fit(
        folds[outer_fold]["outer_train"]["X"],
        folds[outer_fold]["outer_train"]["y"]
    )

    y_pred = final_model.predict(folds[outer_fold]["holdout"]["X"])
    acc = accuracy_score(folds[outer_fold]["holdout"]["y"], y_pred)
    print(f"\nFinal HOLDOUT accuracy (outer fold {outer_fold}): {acc:.4f}")

if __name__ == "__main__":
    main()