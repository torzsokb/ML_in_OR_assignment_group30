import numpy as np
import optuna
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from data_utils import get_folds

folds = get_folds(target_mode=1, scaler_type=None)  
n_folds = len(folds)
current_fold = 0  

def inner_cv(outer_fold: int, C: float, penalty: str, solver: str) -> float:
    inner_losses = []

    for inner_split in folds[outer_fold]["inner_folds"]:

        model = make_pipeline(
            MinMaxScaler(),
            LogisticRegression(
                C=C,
                penalty=penalty,
                solver=solver,
                max_iter=5000
            )
        )

        model.fit(inner_split["train_X"], inner_split["train_y"])
        prob = model.predict_proba(inner_split["test_X"])
        loss = log_loss(inner_split["test_y"], prob)
        inner_losses.append(loss)

    return np.mean(inner_losses)

def objective(trial):
    C = trial.suggest_float("C", 1e-4, 100, log=True)
    penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
    solver = trial.suggest_categorical("solver", ["liblinear", "saga"])

    return inner_cv(outer_fold=current_fold, C=C, penalty=penalty, solver=solver)

if __name__ == "__main__":

    results = []

    for current_fold in range(n_folds):
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=75) 

        best_params = study.best_params
        print(f"Fold {current_fold} best params: {best_params}")

        final_model = make_pipeline(
            MinMaxScaler(),
            LogisticRegression(
                C=best_params["C"],
                penalty=best_params["penalty"],
                solver=best_params["solver"],
                max_iter=5000
            )
        )

        final_model.fit(folds[current_fold]["train_X"], folds[current_fold]["train_y"])

        preds = final_model.predict(folds[current_fold]["holdout_X"])
        probas = final_model.predict_proba(folds[current_fold]["holdout_X"])
        acc = accuracy_score(folds[current_fold]["holdout_y"], preds)
        loss = log_loss(folds[current_fold]["holdout_y"], probas)

        print(f"Fold {current_fold} HOLDOUT Accuracy: {acc:.3f}, Log-loss: {loss:.3f}")

        results.append({"accuracy": acc, "log_loss": loss})

    acc_mean = np.mean([r["accuracy"] for r in results])
    loss_mean = np.mean([r["log_loss"] for r in results])
    print(f"\nOverall HOLDOUT Accuracy: {acc_mean:.3f}, Log-loss: {loss_mean:.3f}")
