import pandas as pd
import numpy as np
import optuna
from optuna.visualization import plot_parallel_coordinate
from plotly.io import show
import json
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("documents/data/processed_data.csv")
df = df.drop(columns=["y2"], axis=1)
# df = df.drop(columns=["x10", "x30", "x39", "x49", "x50", "x52", "x53"], axis=1)

scaler = MinMaxScaler(clip=True)
# df[["x1", "x2", "x3", "x4", "x5"]] = scaler.fit_transform(df[["x1", "x2", "x3", "x4", "x5"]])
n_folds = len(pd.unique(df["cv_fold"]))

folds = {}

for fold in range(n_folds):
    
    fold_numbers = [i for i in range(n_folds)]
    fold_numbers.remove(fold)
    
    train_df = df[df["cv_fold"] != fold]
    scaler.fit(train_df[["x1", "x2", "x3", "x4", "x5"]])
    train_df[["x1", "x2", "x3", "x4", "x5"]] = scaler.transform(train_df[["x1", "x2", "x3", "x4", "x5"]])
    train_X = train_df.drop(columns=["y1", "cv_fold"], axis=1).to_numpy()
    train_y = train_df["y1"].to_numpy()

    holdout_df = df[df["cv_fold"] == fold]
    holdout_df[["x1", "x2", "x3", "x4", "x5"]] = scaler.transform(holdout_df[["x1", "x2", "x3", "x4", "x5"]])
    holdout_X = holdout_df.drop(columns=["y1", "cv_fold"], axis=1).to_numpy()
    holdout_y = holdout_df["y1"].to_numpy()



    inner_folds = []
    for i, inner_fold in enumerate(fold_numbers):
        if not inner_fold == fold:
            
            inner_test_df = train_df[train_df["cv_fold"] == inner_fold]
            inner_test_X = inner_test_df.drop(columns=["y1", "cv_fold"], axis=1).to_numpy()
            inner_test_y = inner_test_df["y1"].to_numpy()
            inner_train_df = train_df[train_df["cv_fold"] != inner_fold]
            inner_train_X = inner_train_df.drop(columns=["y1", "cv_fold"], axis=1).to_numpy()
            inner_train_y = inner_train_df["y1"].to_numpy()
            
            inner_folds.append({"train_X": inner_train_X, "train_y": inner_train_y, "test_X": inner_test_X, "test_y": inner_test_y})

    folds[fold] = {"holdout_X": holdout_X, "holdout_y": holdout_y, "train_X": train_X, "train_y": train_y, "inner_folds": inner_folds} 

k = 0

def next():
    global k
    k += 1

def inner_cv(outer_fold: int, C: float, kernel: str, gamma: float, shrinking: bool, tol: float, max_iter: int, class_weight: str="balanced") -> float:

    scores = []
    for inner_fold in folds[outer_fold]["inner_folds"]:
        model = svm.SVC(C=C, kernel=kernel, gamma=gamma, shrinking=shrinking, tol=tol, max_iter=max_iter, class_weight=class_weight)
        model.fit(inner_fold["train_X"], inner_fold["train_y"])
        preds = model.predict(inner_fold["test_X"])
        scores.append(1-accuracy_score(preds, inner_fold["test_y"]))

    return np.mean(scores)

def objective(trial):


    C = trial.suggest_float("C", 0, 4)
    kernel = trial.suggest_categorical("kernel",["poly", "rbf", "sigmoid"])
    gamma = trial.suggest_float("gamma", 0, 1)
    shrinking = trial.suggest_categorical("shrinking", [False])
    tol = trial.suggest_float("tol", 0.01, 0.5, log=True)
    max_iter = trial.suggest_int("max_iter", 100, 10000, log=True)

    return inner_cv(outer_fold=k, C=C, kernel=kernel, gamma=gamma, shrinking=shrinking, tol=tol, max_iter=max_iter)
    

def main():

    output = {"cv-error": [], "oos-error": []}
    fold_params = []

    for i in range(n_folds):

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0), pruner=optuna.pruners.MedianPruner())
        study.optimize(objective, n_trials=200, timeout=600)
        fig = plot_parallel_coordinate(study)
        show(fig)

        params=study.best_params
        print(params)
        model = svm.SVC(C=params["C"], kernel=params["kernel"], gamma=params["gamma"], shrinking=params["shrinking"], tol=params["tol"], max_iter=params["max_iter"], class_weight="balanced")
        model.fit(folds[k]["train_X"], folds[k]["train_y"])
        preds = model.predict(folds[k]["holdout_X"])
        output["cv-error"].append(study.best_value)
        output["oos-error"].append(1-accuracy_score(preds, folds[k]["holdout_y"]))

        fold_params.append(params)
        with open(f"documents/outputs/xgboost/classification/params/fold_{k}_svm_mms3.json", "w") as f:
            json.dump(params, f)

        next()

    metrics = pd.DataFrame.from_dict(output)
    print(metrics.head(8))
    metrics.to_csv("documents/outputs/xgboost/classification/performance_metrics/out_svm_mms3.csv", index=False)



if __name__ == "__main__":
    main()


