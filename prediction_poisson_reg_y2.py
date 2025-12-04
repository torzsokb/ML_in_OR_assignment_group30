import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance
import optuna
from data_utils import get_folds

folds = get_folds(target_mode=2, scaler_type="minmax")
n_folds = len(folds)

def inner_cv_poiss(outer_fold: int, alpha: float, max_iter: int) -> float:
    inner_scores = []

    for inner_split in folds[outer_fold]["inner_folds"]:
        model = make_pipeline(
            MinMaxScaler(),
            PoissonRegressor(alpha=alpha, max_iter=max_iter, fit_intercept=True)
        )
        model.fit(inner_split["train_X"], inner_split["train_y"])
        preds = model.predict(inner_split["test_X"])
        score = mean_poisson_deviance(inner_split["test_y"], preds)
        inner_scores.append(score)

    return np.mean(inner_scores)

def objective(trial, current_fold: int):
    alpha = trial.suggest_float("alpha", 1e-6, 10, log=True)
    max_iter = trial.suggest_int("max_iter", 100, 5000, log=True)
    return inner_cv_poiss(outer_fold=current_fold, alpha=alpha, max_iter=max_iter)

fold_best_params = []

for current_fold in range(n_folds):
    print(f"Tuning hyperparameters for fold {current_fold}...")
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, current_fold=current_fold), n_trials=100)
    
    fold_best_params.append(study.best_params)
    print(f"Fold {current_fold} best params: {study.best_params}")


alphas = [p["alpha"] for p in fold_best_params]
max_iters = [p["max_iter"] for p in fold_best_params]

best_alpha = float(np.mean(alphas))
best_max_iter = int(np.mean(max_iters))

print(f"Averaged hyperparameters: alpha={best_alpha}, max_iter={best_max_iter}")

full_train_X = np.vstack([folds[f]["train_X"] for f in range(n_folds)])
full_train_y = np.hstack([folds[f]["train_y"] for f in range(n_folds)])

final_model = make_pipeline(
    MinMaxScaler(),
    PoissonRegressor(alpha=best_alpha, max_iter=best_max_iter, fit_intercept=True)
)
final_model.fit(full_train_X, full_train_y)
print("Final Poisson model trained on the entire dataset.")

X_eval_df = pd.read_csv("documents/data/final_preprocessed_evaluation_data.csv")
X_eval_df = X_eval_df.drop(columns=["y1", "y2"])

y_pred = final_model.predict(X_eval_df)
submission = pd.DataFrame({"y2_pred": y_pred})
submission.to_csv("poisson_predictions.csv", index=False)
print("Predictions saved to 'poisson_predictions.csv'.")
