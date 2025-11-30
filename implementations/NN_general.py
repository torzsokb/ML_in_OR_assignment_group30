import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from optuna.visualization import plot_parallel_coordinate
from plotly.io import show
import optuna

import data_utils

NN_mode = 0
NN_layers = 2
class_loss_f = nn.BCEWithLogitsLoss()
reg_loss_f = nn.MSELoss()
mapping = {'Softplus': nn.Softplus(), 'ReLU': nn.ReLU()}
k = 0
folds = None

# Two-layered feed-forward Neural Net class
class NeuralNet(nn.Module):
    def __init__(self,
                 feature_count: int,
                 nodes_layer_one: int,
                 activation_f_one: nn.Module,
                 nodes_layer_two: int = -1,
                 activation_f_two: nn.Module = None):
        super().__init__() # super(RegressionNeuralNet, self).etc by GPT
        # self.flatten = nn.Flatten() # part of pytorch tutorial, not of gpt

        match NN_layers:
            case 1:
                self.NN_body = nn.Sequential(
                nn.Linear(in_features= feature_count, out_features= nodes_layer_one),
                activation_f_one)
                nodes_last_layer = nodes_layer_one

            case _:
                self.NN_body = nn.Sequential(nn.Linear(in_features= feature_count, out_features= nodes_layer_one),
                    activation_f_one,
                    nn.Linear(in_features= nodes_layer_one, out_features= nodes_layer_two),
                    activation_f_two)
                nodes_last_layer = nodes_layer_two

        match NN_mode:
            # Dual
            case 3:
                self.class_head = nn.Linear(in_features= nodes_last_layer, out_features= 1)
                self.reg_head = nn.Linear(in_features= nodes_last_layer, out_features= 1)

            # Bin or reg
            case _:
                self.head = nn.Linear(in_features= nodes_last_layer, out_features= 1)


    def forward(self, x):
        h = self.NN_body(x)
        match NN_mode:
            case 3:
                class_out = self.class_head(h)
                reg_out = self.reg_head(h)

                return class_out, reg_out
            case _:
                return self.head(h)

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
def trained_neural_net(training_data: TensorDataset, nodes_layer_one: int, activation_f_one: nn.Module,
                       learning_rate: float, batch_size: int, stopping_criterion: int, regularization_lambda: float,
                       nodes_layer_two: int = -1, activation_f_two: nn.Module = None):
    feature_count = training_data.tensors[0].shape[1]
    training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

    model = NeuralNet(feature_count,
                      nodes_layer_one, activation_f_one,
                      nodes_layer_two, activation_f_two)

    optimizer = optim.Adam(model.parameters(),
                           lr=learning_rate,
                           weight_decay=regularization_lambda)

    for beta_iter in range(stopping_criterion):
        match NN_mode:
            # Class
            case 1:
                for batch_X, batch_y in training_loader:
                    # Forward pass
                    prediction = model(batch_X)
                    loss = class_loss_f(prediction, batch_y)

                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            # Reg
            case 2:
                for batch_X, batch_y in training_loader:
                    # Forward pass
                    prediction = model(batch_X)
                    loss = reg_loss_f(prediction, batch_y)

                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            # Dual
            case _:
                for batch_X, batch_y_class, batch_y_reg in training_loader:
                    # Forward pass
                    class_pred, reg_pred = model(batch_X)
                    class_loss = class_loss_f(class_pred, batch_y_class)
                    reg_loss = reg_loss_f(reg_pred, batch_y_reg)
                    loss = 10 * class_loss + reg_loss # TODO: tune relative losses!

                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

    model.eval()

    return model

def objective(trial):
    activation_f_one = trial.suggest_categorical('activation_f_one', ['Softplus', 'ReLU'])

    match NN_layers:
        case 1:
            activation_f_two = None
            nodes_layer_two = -1
        case _:
            activation_f_two =  mapping[trial.suggest_categorical('activation_f_two', ['Softplus', 'ReLU'])]
            nodes_layer_two = trial.suggest_int('nodes_layer_two', 3, 30)

    params = {'nodes_layer_one': trial.suggest_int('nodes_layer_one', 3, 30),
              'nodes_layer_two': nodes_layer_two,
              'activation_f_one': mapping[activation_f_one], # nn.Softplus(), nn.ReLU()
              'activation_f_two': activation_f_two,
              'learning_rate': trial.suggest_float('learning_rate', 0.0001, 1, log=True),
              'batch_size': trial.suggest_int('batch_size', 3, 100),
              'stopping_criterion': trial.suggest_int('stopping_criterion', 100, 500),
              'regularization_lambda': trial.suggest_float('regularization_lambda', 1e-4, 50, log=True)
    }

    return inner_cv(outer_fold=k, params=params)

def inner_cv(outer_fold: int, params: dict):
    inner_mse = []

    for inner_split in folds[outer_fold]["inner_folds"]:
        train_x = torch.from_numpy(inner_split["train_X"]).float()
        match NN_mode:
            case 3:
                # column 0 and 1 correspond to binary and reg respectively
                train_y_class = torch.from_numpy(inner_split["train_y"][:, 0]).unsqueeze(1).float()
                train_y_reg = torch.from_numpy(inner_split["train_y"][:, 1]).unsqueeze(1).float()
                training_data = TensorDataset(train_x, train_y_class, train_y_reg)
            case _:
                train_y = torch.from_numpy(inner_split["train_y"]).unsqueeze(1).float()
                training_data = TensorDataset(train_x, train_y)

        model = trained_neural_net(training_data=training_data,
                                   nodes_layer_one= params['nodes_layer_one'],
                                   nodes_layer_two=params['nodes_layer_two'],
                                   activation_f_one=params['activation_f_one'],
                                   activation_f_two=params['activation_f_two'],
                                   learning_rate=params['learning_rate'],
                                   batch_size=params['batch_size'],
                                   stopping_criterion=params['stopping_criterion'],
                                   regularization_lambda=params['regularization_lambda']
                                   )

        test_X = torch.from_numpy(inner_split["test_X"]).float()
        with torch.no_grad():
            match NN_mode:
                case 1:
                    test_y = torch.from_numpy(inner_split["test_y"]).unsqueeze(1).float()
                    predictions = model(test_X)
                    loss = class_loss_f(predictions, test_y)
                case 2:
                    test_y = torch.from_numpy(inner_split["test_y"]).unsqueeze(1).float()
                    predictions = model(test_X)
                    loss = reg_loss_f(predictions, test_y)
                case _:
                    # column 0 and 1 correspond to binary and reg respectively
                    test_y_class = torch.from_numpy(inner_split["test_y"][:, 0]).unsqueeze(1).float()
                    test_y_reg = torch.from_numpy(inner_split["test_y"][:, 0]).unsqueeze(1).float()
                    class_pred, reg_pred = model(test_X)

                    class_loss = class_loss_f(class_pred, test_y_class)
                    reg_loss = reg_loss_f(reg_pred, test_y_reg)
                    loss = 10 * class_loss + reg_loss # TODO: tune relative losses!
            inner_mse.append(loss.item())

    return np.mean(inner_mse)

def run_NN_algorithm(target_mode: int, target_layers: int, input_folds: dict):
    """ target_mode
                        1: y1 (binary)
                        2: y2 (regression)
                        3: [y1, y2] (simultaneous)
                        o/w: error is thrown
            target_layers
                        number of layers in the neural net (either 1 or 2)
                        other values result in 2"""
    global NN_mode, NN_layers, folds
    NN_mode = target_mode
    NN_layers = target_layers
    if NN_mode not in (1, 2, 3):
        raise ValueError(f"Invalid input: {NN_mode}")

    if NN_layers not in (1,2):
        print(f"Invalid layer count: {NN_layers}, should be 1 or 2")

    folds = input_folds
    n_folds = len(folds)

    output = {"oos_rmse": [], "oos_class_acc": [], "oos_class_loss_f": [], "oos_dual_loss": []}

    for fold in range(n_folds):
        global k
        k = fold

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=100, timeout=900)
        fig = plot_parallel_coordinate(study)
        show(fig)

        best_params = study.best_params
        print(f"Fold {fold} best params: {best_params}")

        # Full fold training data
        ff_train_X = torch.from_numpy(folds[fold]["train_X"]).float()
        match NN_mode:
            case 3:
                # column 0 and 1 correspond to binary and reg respectively
                ff_train_y_class = torch.from_numpy(folds[fold]["train_y"][:, 0]).unsqueeze(1).float()
                ff_train_y_reg = torch.from_numpy(folds[fold]["train_y"][:, 1]).unsqueeze(1).float()
                ff_training_data = TensorDataset(ff_train_X, ff_train_y_class, ff_train_y_reg)
            case _:
                ff_train_y = torch.from_numpy(folds[fold]["train_y"]).unsqueeze(1).float()
                ff_training_data = TensorDataset(ff_train_X, ff_train_y)

        match NN_layers:
            case 1:
                activation_f_two = None
                nodes_layer_two = -1
            case _:
                activation_f_two = mapping[best_params['activation_f_two']]
                nodes_layer_two = best_params['nodes_layer_two']

        final_model = trained_neural_net(training_data=ff_training_data,
                                         nodes_layer_one=best_params['nodes_layer_one'],
                                         activation_f_one=mapping[best_params['activation_f_one']],
                                         nodes_layer_two=nodes_layer_two,
                                         activation_f_two=activation_f_two,
                                         learning_rate=best_params['learning_rate'],
                                         batch_size=best_params['batch_size'],
                                         stopping_criterion=best_params['stopping_criterion'],
                                         regularization_lambda=best_params['regularization_lambda']
                                         )

        holdout_X = torch.from_numpy(folds[fold]["holdout_X"]).float()
        match NN_mode:
            case 1:
                holdout_y = torch.from_numpy(folds[fold]["holdout_y"]).unsqueeze(1).float()
                with torch.no_grad():
                    fold_predictions = final_model(holdout_X)
                    fold_loss = class_loss_f(fold_predictions, holdout_y)

                    probabilities = torch.sigmoid(fold_loss)
                    class_oos_predictions = (probabilities > 0.5).long()
                    class_oos_true = holdout_y.long()

                    output["oos_class_acc"].append(class_oos_predictions == class_oos_true)
                    output["oos_class_loss_f"].append(fold_loss.item())

                print(f"Fold {fold} holdout binary accuracy: {(class_oos_predictions == class_oos_true).sum().item() / holdout_y.size(0):.4f}")

            case 2:
                holdout_y = torch.from_numpy(folds[fold]["holdout_y"]).unsqueeze(1).float()
                with torch.no_grad():
                    fold_predictions = final_model(holdout_X)
                    fold_loss = reg_loss_f(fold_predictions, holdout_y)
                    output["oos_rmse"].append(fold_loss.item())

            case _:
                # column 0 and 1 correspond to binary and reg respectively
                holdout_y_class = torch.from_numpy(folds[fold]["holdout_y"][:, 0]).unsqueeze(1).float()
                holdout_y_reg = torch.from_numpy(folds[fold]["holdout_y"][:, 0]).unsqueeze(1).float()
                with torch.no_grad():
                    class_fold_predictions, reg_fold_predictions = final_model(holdout_X)
                    reg_loss = reg_loss_f(reg_fold_predictions, holdout_y_reg)
                    class_loss = class_loss_f(class_fold_predictions, holdout_y_class)
                    fold_loss = class_loss + reg_loss

                    probabilities = torch.sigmoid(class_loss)
                    class_oos_predictions = (probabilities > 0.5).long()
                    class_oos_true = holdout_y_class.long()

                    output["oos_class_acc"].append(class_oos_predictions == class_oos_true)
                    output["oos_rmse"].append(reg_loss.item())
                    output["oos_dual_loss"].append(fold_loss.item())

                print(f"Fold {fold} holdout binary accuracy: {(class_oos_predictions == class_oos_true).sum().item() / holdout_y_class.size(0):.4f}")
                print(f"Fold {fold} holdout regression mse: {reg_loss.item():.4f}")

        print(f"Fold {fold} HOLDOUT loss: {fold_loss:.4f}")

    match NN_mode:
        case 1:
            average_accuracy = np.mean(output["oos_class_acc"])
            print(f"\nOverall HOLDOUT accuracy: {average_accuracy:.4f}")

        case 2:
            mse_mean = np.mean(output["oos_rmse"])
            print(f"\nOverall HOLDOUT MSE: {mse_mean:.4f}")

        case _:
            mse_mean_reg = np.mean(output["oos_rmse"])
            average_accuracy = np.mean(output["oos_class_acc"])
            print(f"\nRegression HOLDOUT MSE: {mse_mean_reg:.4f}")
            print(f"\nClassification HOLDOUT accuracy: {average_accuracy:.4f}")


if __name__ == "__main__":
    target_mode = 1
    target_layers = 1

    input_folds = data_utils.get_folds(target_mode, 'minmax')
    print("Data read")

    run_NN_algorithm(target_mode, target_layers, input_folds=input_folds)
