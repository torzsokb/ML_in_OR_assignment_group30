import numpy as np
import optuna
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_squared_error, mean_poisson_deviance
from data_utils import get_folds 

folds = get_folds(target_mode=2, scaler_type=None)
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

# Objective for Optuna
def objective(trial):
    alpha = trial.suggest_float("alpha", 1e-6, 10, log=True)
    max_iter = trial.suggest_int("max_iter", 100, 5000, log=True)
    return inner_cv_poiss(outer_fold=current_fold, alpha=alpha, max_iter=max_iter)

if __name__ == "__main__":

    results = []
    for current_fold in range(n_folds):

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=75)

        best_params = study.best_params
        print(f"Fold {current_fold} best params: {best_params}")

        # Final model
        final_model = make_pipeline(
            MinMaxScaler(),
            PoissonRegressor(
                alpha=best_params["alpha"],
                max_iter=best_params["max_iter"],
                fit_intercept=True
            )
        )

        final_model.fit(folds[current_fold]["train_X"], folds[current_fold]["train_y"])

        train_preds = final_model.predict(folds[current_fold]["train_X"])

        oos_preds = final_model.predict(folds[current_fold]["holdout_X"])

        cv_poiss = inner_cv_poiss(current_fold, best_params["alpha"], best_params["max_iter"])

        train_mse = mean_squared_error(folds[current_fold]["train_y"], train_preds)
        oos_mse = mean_squared_error(folds[current_fold]["holdout_y"], oos_preds)

        train_poiss = mean_poisson_deviance(folds[current_fold]["train_y"], train_preds)
        oos_poiss = mean_poisson_deviance(folds[current_fold]["holdout_y"], oos_preds)

        print(f"Fold {current_fold} HOLDOUT MSE: {oos_mse:.4f}, Poisson deviance: {oos_poiss:.4f}")

        results.append({
            "fold": current_fold,
            "cv-poiss-dev": cv_poiss,
            "train-poiss-dev": train_poiss,
            "oos-poiss-dev": oos_poiss,
            "train-mse": train_mse,
            "oos-mse": oos_mse,
            "best_params": best_params
        })

    print("\nfold  cv-poiss-dev  train-poiss-dev  oos-poiss-dev  train-mse    oos-mse")
    for r in results:
        print(f"{r['fold']:>2}    {r['cv-poiss-dev']:.6f}       {r['train-poiss-dev']:.6f}        "
              f"{r['oos-poiss-dev']:.6f}      {r['train-mse']:.6f}   {r['oos-mse']:.6f}")

    print("\nOverall means:")
    print("cv-poiss-dev   =", np.mean([r["cv-poiss-dev"] for r in results]))
    print("train-poiss-dev=", np.mean([r["train-poiss-dev"] for r in results]))
    print("oos-poiss-dev  =", np.mean([r["oos-poiss-dev"] for r in results]))
    print("train-mse      =", np.mean([r["train-mse"] for r in results]))
    print("oos-mse        =", np.mean([r["oos-mse"] for r in results]))

    # Extract alphas and max_iters from all folds
    alphas = [r["best_params"]["alpha"] for r in results]
    max_iters = [r["best_params"]["max_iter"] for r in results]

    best_alpha = float(np.mean(alphas))
    best_max_iter = int(np.mean(max_iters))

    print(f"Averaged hyperparameters across folds: alpha={best_alpha}, max_iter={best_max_iter}")