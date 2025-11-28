import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from optuna.visualization import plot_parallel_coordinate
from plotly.io import show
import optuna

import data_utils

# Two-layered feed-forward Neural Net class
class RegressionNeuralNet(nn.Module):
    def __init__(self,
                 feature_count: int,
                 nodes_layer_one: int,
                 activation_f_one: nn.Module,
                 nodes_layer_two: int,
                 activation_f_two: nn.Module):
        super().__init__() # super(RegressionNeuralNet, self).etc by GPT
        # self.flatten = nn.Flatten() # part of pytorch tutorial, not of gpt
        self.net = nn.Sequential(
            nn.Linear(in_features= feature_count, out_features= nodes_layer_one),
            activation_f_one,
            nn.Linear(in_features= nodes_layer_one, out_features= nodes_layer_two),
            activation_f_two,
            nn.Linear(in_features= nodes_layer_two, out_features= 1)
        )

    def forward(self, x):
        return self.net(x)

# Creates and trains the neural net
#
# Hyperparameters to be tuned / to be chosen:
# x First layer:
# x   # of nodes
# x   Activation function
# x Second layer:
# x   # of nodes
# x   Activation function
# ? Starting values for beta
# ? Loss function (i.e.: nn.MSELoss() )
# x Learning rate
# x Mini-batch subgradient descent: batch size
# ? Stopping criterion (currently beta update iteration count, also need to consider stopping crit hyper params)
# x Regularization parameter \lambda (currently lambda for ridge, gpt recommended, gpt also recommended dropout)
def trained_neural_net(training_data: TensorDataset, nodes_layer_one: int, nodes_layer_two: int,
                       activation_f_one: nn.Module, activation_f_two: nn.Module, loss_f: nn.Module, learning_rate: float,
                       batch_size: int, stopping_criterion: int, regularization_lambda: float):
    feature_count = training_data.tensors[0].shape[1]
    training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

    model = RegressionNeuralNet(feature_count,
                                nodes_layer_one, activation_f_one,
                                nodes_layer_two, activation_f_two)

    optimizer = optim.Adam(model.parameters(),
                           lr=learning_rate,
                           weight_decay=regularization_lambda)

    for beta_iter in range(stopping_criterion):
        for batch_X, batch_y in training_loader:
            # Forward pass
            prediction = model(batch_X)
            loss = loss_f(prediction, batch_y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()

    return model

def objective(trial):
    params = {
        'nodes_layer_one': trial.suggest_int('nodes_layer_one', 1, 50),
        'nodes_layer_two': trial.suggest_int('nodes_layer_two', 1, 50),
        'activation_f_one': trial.suggest_categorical('activation_f_one', [nn.Softplus(), nn.ReLU()]), # nn.Softplus()
        'activation_f_two': trial.suggest_categorical('activation_f_two', [nn.Softplus(), nn.ReLU()]),
        'learning_rate': trial.suggest_float('learning_rate', 0.0001, 1, log=True),
        'batch_size': trial.suggest_int('batch_size', 3, 100),
        'stopping_criterion': trial.suggest_int('stopping_criterion', 100, 1000),
        'regularization_lambda': trial.suggest_float('regularization_lambda', 1e-4, 10, log=True)
    }

    # loss_f = nn.MSELoss()

    return inner_cv(outer_fold=fold, params=params, loss_f=global_loss_f)

def inner_cv(outer_fold: int, params: dict, loss_f: nn.Module):
    inner_mse = []

    for inner_split in folds[outer_fold]["inner_folds"]:
        train_x = torch.tensor(inner_split["train_X"].values, dtype=torch.float32)
        train_y = torch.tensor(inner_split["train_y"].values, dtype=torch.float32).unsqueeze(1)
        training_data = TensorDataset(train_x, train_y)

        model = trained_neural_net(training_data=training_data,
                                   nodes_layer_one= params['nodes_layer_one'],
                                   activation_f_one=params['activation_f_one'],
                                   nodes_layer_two=params['nodes_layer_two'],
                                   activation_f_two=params['activation_f_two'],
                                   loss_f=loss_f,
                                   learning_rate=params['learning_rate'],
                                   batch_size=params['batch_size'],
                                   stopping_criterion=params['stopping_criterion'],
                                   regularization_lambda=params['regularization_lambda']
                                   )

        test_x = torch.tensor(inner_split["test_X"].values, dtype=torch.float32)
        test_y = torch.tensor(inner_split["test_y"].values, dtype=torch.float32).unsqueeze(1)
        with torch.no_grad():
            preds = model(test_x)
            loss = loss_f(preds, test_y)
            inner_mse.append(loss.item())

    return np.mean(inner_mse)


if __name__ == "__main__":
    global_loss_f = nn.MSELoss()

    folds = data_utils.get_folds(2, 'minmax')
    n_folds = len(folds)
    print("Data read")

    output = {"train-rmse": [], "val-rmse": [], "oos-rmse": []}

    for fold in range(n_folds):
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=100, timeout=600)
        fig = plot_parallel_coordinate(study)
        show(fig)

        best_params = study.best_params
        print(f"Fold {fold} best params: {best_params}")

        # Full fold training data
        ff_train_X = torch.tensor(folds[fold]["train_X"].values, dtype=torch.float32)
        ff_train_y = torch.tensor(folds[fold]["train_y"].values, dtype=torch.float32).unsqueeze(1)
        ff_training_data = TensorDataset(ff_train_X, ff_train_y)

        final_model = trained_neural_net(training_data=ff_training_data,
                                         loss_f=global_loss_f,
                                         nodes_layer_one=best_params['nodes_layer_one'],
                                         activation_f_one=best_params['activation_f_one'],
                                         nodes_layer_two=best_params['nodes_layer_two'],
                                         activation_f_two=best_params['activation_f_two'],
                                         learning_rate=best_params['learning_rate'],
                                         batch_size=best_params['batch_size'],
                                         stopping_criterion=best_params['stopping_criterion'],
                                         regularization_lambda=best_params['regularization_lambda']
                                         )

        holdout_X = torch.tensor(folds[fold]["holdout_X"].values, dtype=torch.float32)
        holdout_y = torch.tensor(folds[fold]["holdout_y"].values, dtype=torch.float32).unsqueeze(1)

        with torch.no_grad():
            fold_preds = final_model(holdout_X)
            fold_loss = global_loss_f(fold_preds, holdout_y)
            output["oos-rmse"].append(fold_loss.item())

        print(f"Fold {fold} HOLDOUT MSE: {fold_loss:.4f}")

    mse_mean = np.mean(output["oos-rmse"])
    print(f"\nOverall HOLDOUT MSE: {mse_mean:.4f}")