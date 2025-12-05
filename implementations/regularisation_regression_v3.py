import numpy as np
import optuna
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from data_utils import get_folds
import pandas as pd

folds = get_folds(target_mode=2, scaler_type=None)
n_folds = len(folds)

df = pd.read_csv("documents/data/processed_data.csv")
print(df.mean())

def inner_cv_mse(outer_fold: int, alpha: float, l1_ratio: float) -> float:
    inner_mse = []

    for inner_split in folds[outer_fold]["inner_folds"]:
        model = make_pipeline(
            MinMaxScaler(),
            ElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                max_iter=5000,
                random_state=42
            )
        )

        model.fit(inner_split["train_X"], inner_split["train_y"])
        preds = model.predict(inner_split["test_X"])
        mse = mean_squared_error(inner_split["test_y"], preds)
        inner_mse.append(mse)

    return np.mean(inner_mse)

# Optuna objective --------------------------------
def objective(trial):
    alpha = trial.suggest_float("alpha", 1e-4, 10, log=True)
    l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)

    return inner_cv_mse(
        outer_fold=current_fold,
        alpha=alpha,
        l1_ratio=l1_ratio
    )

if __name__ == "__main__":

    results = []

    for current_fold in range(n_folds):
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=75)

        best_params = study.best_params
        print(f"Fold {current_fold} best params: {best_params}")

        # Train final model
        final_model = make_pipeline(
            MinMaxScaler(),
            ElasticNet(
                alpha=best_params["alpha"],
                l1_ratio=best_params["l1_ratio"],
                max_iter=5000,
                random_state=42
            )
        )

        final_model.fit(folds[current_fold]["train_X"], folds[current_fold]["train_y"])

        # Predictions
        train_preds = final_model.predict(folds[current_fold]["train_X"])
        oos_preds = final_model.predict(folds[current_fold]["holdout_X"])

        # Compute metrics
        cv_mse = inner_cv_mse(current_fold, best_params["alpha"], best_params["l1_ratio"])
        train_mse = mean_squared_error(folds[current_fold]["train_y"], train_preds)
        oos_mse = mean_squared_error(folds[current_fold]["holdout_y"], oos_preds)

        print(f"Fold {current_fold} HOLDOUT MSE: {oos_mse:.4f}")

        results.append({
            "fold": current_fold,
            "cv-mse": cv_mse,
            "train-mse": train_mse,
            "oos-mse": oos_mse
        })

    print("\nfold   cv-mse      train-mse    oos-mse")
    for r in results:
        print(f"{r['fold']:>2}   {r['cv-mse']:.6f}   {r['train-mse']:.6f}   {r['oos-mse']:.6f}")

    mse_mean = np.mean([r["oos-mse"] for r in results])
    print(f"\nOverall HOLDOUT MSE: {mse_mean:.6f}")
