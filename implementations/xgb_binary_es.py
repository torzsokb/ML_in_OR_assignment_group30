import pandas as pd
import xgboost as xgb
import numpy as np
import optuna
from optuna.visualization import plot_parallel_coordinate
from plotly.io import show
import json

df = pd.read_csv("documents/data/processed_data.csv")
df = df.drop(columns=["y2"], axis=1)
n_folds = len(pd.unique(df["cv_fold"]))

folds = {}
num_boost_round = 6


for fold in range(n_folds):
    
    fold_numbers = [i for i in range(n_folds)]
    fold_numbers.remove(fold)
    outer_validation = (fold + 1) % n_folds
    print(f"fold: {fold}, outer_validation: {outer_validation}")
    holdout_df = df[df["cv_fold"] == fold]
    holdout_data = xgb.DMatrix(data = holdout_df.drop(["y1", "cv_fold"], axis=1), label=holdout_df["y1"])

    train_df = df[df["cv_fold"] != fold]
    outer_train_df_split = train_df[train_df["cv_fold"] != outer_validation]
    outer_train_data = xgb.DMatrix(data = outer_train_df_split.drop(["y1", "cv_fold"], axis=1), label=outer_train_df_split["y1"])
    outer_validation_df = train_df[train_df["cv_fold"] == outer_validation]
    outer_validation_data = xgb.DMatrix(data = outer_validation_df.drop(["y1", "cv_fold"], axis=1), label=outer_validation_df["y1"])

    inner_folds = []
    for i, inner_fold in enumerate(fold_numbers):
        if not inner_fold == fold:
            
            inner_test_df = train_df[train_df["cv_fold"] == inner_fold]
            inner_train_df = train_df[train_df["cv_fold"] != inner_fold]

            validation = fold_numbers[(i + fold + 1) % (n_folds - 1)]
            if validation == inner_fold:
                validation = fold_numbers[(i + fold + 2) % (n_folds - 1)]

            print(f"\tinner fold: {inner_fold}, inner validation: {validation}")
            inner_train_df_split = inner_train_df[inner_train_df["cv_fold"] != validation]
            inner_validation_df = inner_train_df[inner_train_df["cv_fold"] == validation]

            inner_test_data = xgb.DMatrix(data = inner_test_df.drop(["y1", "cv_fold"], axis=1), label=inner_test_df["y1"])
            inner_train_data = xgb.DMatrix(data = inner_train_df_split.drop(["y1", "cv_fold"], axis=1), label=inner_train_df_split["y1"])
            inner_validation_data = xgb.DMatrix(data = inner_validation_df.drop(["y1", "cv_fold"], axis=1), label=inner_validation_df["y1"])
            
            inner_folds.append({"train": inner_train_data, "test": inner_test_data, "validation": inner_validation_data, "balance": (1 - inner_train_df_split["y1"].mean()) / inner_train_df_split["y1"].mean()})

    folds[fold] = {"holdout": holdout_data, "train": outer_train_data, "validation": outer_validation_data,"inner_folds": inner_folds, "balance": (1 - outer_train_df_split["y1"].mean()) / outer_train_df_split["y1"].mean()} 

k = 0

def next():
    global k
    k += 1

def inner_cv(outer_fold: int, params: dict, num_boost_round: int, early_stopping_rounds: int, min_delta: float) -> float:

    scores = []
    for inner_fold in folds[outer_fold]["inner_folds"]:
        evals_result = {}
        # params["scale_pos_weight"] = inner_fold["balance"]
        # params["num_class"] = 1
        es = xgb.callback.EarlyStopping(rounds=early_stopping_rounds, metric_name="error", data_name="val", save_best=True, maximize=False, min_delta=min_delta)
        model = xgb.train(params, num_boost_round=num_boost_round,
                          dtrain=inner_fold["train"], 
                          evals=[(inner_fold["test"], "test"), (inner_fold["train"], "train"), (inner_fold["validation"], "val")], 
                          callbacks = [es],
                          evals_result=evals_result, verbose_eval=False)
        
        scores.append(evals_result["test"]["error"][model.best_iteration])

    return np.mean(scores)

def objective(trial):

    params = {'learning_rate': trial.suggest_float('learning_rate', 0.0001, 1, log=True),
              'max_depth': trial.suggest_int('max_depth', 1, 10),
              'min_child_weight': trial.suggest_int('min_child_weight', 1, 12),
              'gamma': trial.suggest_float('gamma', 0.1, 10),
              'reg_lambda': trial.suggest_float('reg_lambda', 0, 7),
              'reg_alpha': trial.suggest_float('reg_alpha', 0, 7),
              'max_bin': trial.suggest_categorical('max_bin', [16, 32, 64, 128, 256, 512]),
              'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1),
              'colsample_bylevel' : trial.suggest_float('colsample_bylevel', 0.1, 1),
              'colsample_bynode': trial.suggest_float('colsample_bynode', 0.1, 1),
              'subsample': trial.suggest_float('subsample', 0.4, 1),
              'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
              'refresh_leaf': trial.suggest_categorical('refresh_leaf', [0, 1]),
              'num_parallel_tree': trial.suggest_int('num_parallel_tree', 1, 15),
              'random_state': 0,
              'objective': trial.suggest_categorical('objective', ["binary:logistic"]),
              'eval_metric': ["error", "logloss"]
              }

    early_stopping_rounds = trial.suggest_int('early_stopping_rounds', 2, 5)
    min_delta = trial.suggest_float('min_delta', 0.000001, 0.2, log=True)
    return inner_cv(outer_fold=k, params=params, num_boost_round=num_boost_round, early_stopping_rounds=early_stopping_rounds, min_delta=min_delta)
    

def main():

    output = {"cv-error": [], "train-error": [], "val-error": [], "oos-error": [], "train-logloss": [], "val-logloss": [], "oos-logloss": []}
    fold_params = {}
    feature_importances = {}

    for i in range(n_folds):

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0), pruner=optuna.pruners.MedianPruner())
        study.optimize(objective, n_trials=150, timeout=600)
        fig = plot_parallel_coordinate(study)
        show(fig)

        final_evals_result = {}
        params=study.best_params
        print(params)
        params["eval_metric"] = ["error", "logloss"]
        # params["num_class"] = 1
        # params["scale_pos_weight"]=folds[k]["balance"]
        es = xgb.callback.EarlyStopping(rounds=study.best_params["early_stopping_rounds"], metric_name="error", data_name="val", save_best=True, maximize=False, min_delta=study.best_params["min_delta"])
        model = xgb.train(params, num_boost_round=num_boost_round, 
                          dtrain=folds[k]["train"], 
                          evals=[(folds[k]["holdout"], "oos"), (folds[k]["train"], "train"), (folds[k]["validation"], "val")],
                          callbacks = [es],
                          evals_result=final_evals_result, verbose_eval=True)
        
        best_idx = model.best_iteration
        output["cv-error"].append(study.best_value)
        output["train-error"].append(final_evals_result["train"]["error"][best_idx])
        output["val-error"].append(final_evals_result["val"]["error"][best_idx])
        output["oos-error"].append(final_evals_result["oos"]["error"][best_idx])

        output["train-logloss"].append(final_evals_result["train"]["logloss"][best_idx])
        output["val-logloss"].append(final_evals_result["val"]["logloss"][best_idx])
        output["oos-logloss"].append(final_evals_result["oos"]["logloss"][best_idx])

        importance_gain  = model.get_score(importance_type='gain')
        importance_weight = model.get_score(importance_type='weight')

        fold_params[k] = params
        feature_importances[k] = {
            'feature': list(importance_gain.keys()),
            'gain': list(importance_gain.values()),
            'weight': [importance_weight.get(f, 0) for f in importance_gain.keys()]
            }
        

        next()
    with open(f"documents/outputs/xgboost/classification/params/xgb_es_r{num_boost_round}.json", "w") as f:
            json.dump(fold_params, f)
    with open(f"documents/outputs/xgboost/classification/performance_metrics/xgb_es_r{num_boost_round}.json", "w") as f:
            json.dump(feature_importances, f)
    metrics = pd.DataFrame.from_dict(output)
    print(metrics.head(8))
    metrics.to_csv(f"documents/outputs/xgboost/classification/performance_metrics/out_xgb_es_r{num_boost_round}.csv", index=False)



if __name__ == "__main__":
    main()


