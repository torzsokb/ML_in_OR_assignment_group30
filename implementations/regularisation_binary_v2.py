import pandas as pd
import numpy as np
import optuna

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, log_loss

df = pd.read_csv("documents/data/processed_data.csv")
df = df.drop(columns=["y2"], axis=1)   

y_name = "y1"                          
n_folds = len(pd.unique(df["cv_fold"]))

folds = {}

for fold in range(n_folds):
    
    fold_numbers = list(range(n_folds))
    fold_numbers.remove(fold)
    outer_validation = fold + 1 if fold + 1 < n_folds else 0

    print(f"fold: {fold}, outer_validation: {outer_validation}")

    holdout_df = df[df["cv_fold"] == fold]
    holdout_data = {
        "X": holdout_df.drop([y_name, "cv_fold"], axis=1).values,
        "y": holdout_df[y_name].values
    }


    train_df = df[df["cv_fold"] != fold]
    outer_train_df_split = train_df[train_df["cv_fold"] != outer_validation]
    outer_train_data = {
        "X": outer_train_df_split.drop([y_name, "cv_fold"], axis=1).values,
        "y": outer_train_df_split[y_name].values
    }

    outer_validation_df = train_df[train_df["cv_fold"] == outer_validation]
    outer_validation_data = {
        "X": outer_validation_df.drop([y_name, "cv_fold"], axis=1).values,
        "y": outer_validation_df[y_name].values
    }

    inner_folds = []
    for i, inner_fold in enumerate(fold_numbers):

        inner_test_df = train_df[train_df["cv_fold"] == inner_fold]
        inner_train_df = train_df[train_df["cv_fold"] != inner_fold]

        validation = fold_numbers[0] if i + 1 == len(fold_numbers) else fold_numbers[i+1]

        inner_train_df_split = inner_train_df[inner_train_df["cv_fold"] != validation]
        inner_validation_df = inner_train_df[inner_train_df["cv_fold"] == validation]

        inner_folds.append({
            "train": {
                "X": inner_train_df_split.drop([y_name, "cv_fold"], axis=1).values,
                "y": inner_train_df_split[y_name].values
            },
            "test": {
                "X": inner_test_df.drop([y_name, "cv_fold"], axis=1).values,
                "y": inner_test_df[y_name].values
            },
            "validation": {
                "X": inner_validation_df.drop([y_name, "cv_fold"], axis=1).values,
                "y": inner_validation_df[y_name].values
            }
        })

    folds[fold] = {
        "holdout": holdout_data,
        "train": outer_train_data,
        "validation": outer_validation_data,
        "inner_folds": inner_folds
    }

def inner_cv(outer_fold, params):

    scores = []

    for inner_fold in folds[outer_fold]["inner_folds"]:

        model = make_pipeline(
            MinMaxScaler(),
            LogisticRegression(
                C=params["C"],
                penalty=params["penalty"],
                solver=params["solver"],
                max_iter=5000
            )
        )

        model.fit(inner_fold["train"]["X"], inner_fold["train"]["y"])

        prob = model.predict_proba(inner_fold["test"]["X"])
        loss = log_loss(inner_fold["test"]["y"], prob) 

        scores.append(loss)

    return np.mean(scores)

k = 0 

def objective(trial):

    params = {
        "C": trial.suggest_float("C", 1e-4, 100, log=True),       
        "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
        "solver": trial.suggest_categorical("solver", ["liblinear", "saga"])  
    }

    return inner_cv(k, params)


def main():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=75)

    best_params = study.best_params
    print("Best params:", best_params)

    final_model = make_pipeline(
        MinMaxScaler(),
        LogisticRegression(
            C=best_params["C"],
            penalty=best_params["penalty"],
            solver=best_params["solver"],
            max_iter=5000
        )
    )

    final_model.fit(folds[k]["train"]["X"], folds[k]["train"]["y"])

    y_pred = final_model.predict(folds[k]["holdout"]["X"])
    acc = accuracy_score(folds[k]["holdout"]["y"], y_pred)

    print(f"\nFinal HOLDOUT accuracy (outer fold {k}): {acc:.4f}")


if __name__ == "__main__":
    main()
