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

def classification_error(y_true, y_pred):
    return 1 - accuracy_score(y_true, y_pred)

def inner_cv_error(outer_fold, C, penalty, solver):
    errors = []
    for inner_split in folds[outer_fold]["inner_folds"]:
        model = make_pipeline(
            MinMaxScaler(),
            LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=5000)
        )
        model.fit(inner_split["train_X"], inner_split["train_y"])
        pred = model.predict(inner_split["test_X"])
        errors.append(classification_error(inner_split["test_y"], pred))
    return np.mean(errors)


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
        
        # compute cv-error using the best params
        cv_err = inner_cv_error(current_fold, best_params["C"], best_params["penalty"], best_params["solver"])

        # training predictions
        train_pred = final_model.predict(folds[current_fold]["train_X"])
        train_prob = final_model.predict_proba(folds[current_fold]["train_X"])

        train_err = classification_error(folds[current_fold]["train_y"], train_pred)
        train_ll = log_loss(folds[current_fold]["train_y"], train_prob)

        # out-of-sample = holdout
        oos_err = classification_error(folds[current_fold]["holdout_y"], preds)
        oos_ll = loss

        fold_results = {
            "fold": current_fold,
            "cv-error": cv_err,
            "train-error": train_err,
            "oos-error": oos_err,
            "train-logloss": train_ll,
            "oos-logloss": oos_ll
        }

        results.append(fold_results)

    print("\nfold  cv-error  train-error  oos-error  train-logloss  oos-logloss")
    for r in results:
        print(f"{r['fold']:>2}   {r['cv-error']:.6f}   {r['train-error']:.6f}   "
          f"{r['oos-error']:.6f}   {r['train-logloss']:.6f}   {r['oos-logloss']:.6f}")

