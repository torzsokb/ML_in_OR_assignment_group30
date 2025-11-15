import numpy as np
import optuna
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_squared_error, mean_poisson_deviance
from data_utils import get_folds 

folds = get_folds(target_mode=2, scaler_type=None)
n_folds = len(folds)

def inner_cv(outer_fold: int, alpha: float, max_iter: int) -> float:

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

def objective(trial):
    alpha = trial.suggest_float("alpha", 1e-6, 10, log=True)
    max_iter = trial.suggest_int("max_iter", 100, 5000, log=True)

    return inner_cv(outer_fold=current_fold, alpha=alpha, max_iter=max_iter)

if __name__ == "__main__":

    results = []

    for current_fold in range(n_folds):

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=75)  

        best_params = study.best_params
        print(f"Fold {current_fold} best params: {best_params}")

        final_model = make_pipeline(
            MinMaxScaler(),
            PoissonRegressor(alpha=best_params["alpha"],
                             max_iter=best_params["max_iter"],
                             fit_intercept=True)
        )

        final_model.fit(folds[current_fold]["train_X"], folds[current_fold]["train_y"])

        preds = final_model.predict(folds[current_fold]["holdout_X"])
        mse = mean_squared_error(folds[current_fold]["holdout_y"], preds)
        poisson_dev = mean_poisson_deviance(folds[current_fold]["holdout_y"], preds)

        print(f"Fold {current_fold} HOLDOUT MSE: {mse:.4f}, Poisson deviance: {poisson_dev:.4f}")
        results.append({"mse": mse, "poisson_dev": poisson_dev})


    mse_mean = np.mean([r["mse"] for r in results])
    poisson_dev_mean = np.mean([r["poisson_dev"] for r in results])
    print(f"\nOverall HOLDOUT MSE: {mse_mean:.4f}, Poisson deviance: {poisson_dev_mean:.4f}")
