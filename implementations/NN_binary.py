import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# Differences with NN_regression:
# loss function
# Data requested from data_utils


def main():
    df = pd.read_csv(r"C:\Users\roela\PycharmProjects\ML_in_OR_assignment_group30\documents\data\processed_data.csv")
    df = df.drop(columns=["y2"], axis=1) # Remove regression data
    n_folds = len(pd.unique(df["cv_fold"]))

    in_sample_losses = []
    oos_losses = []

    for fold in range(n_folds):
        fold_numbers = [i for i in range(n_folds)]
        fold_numbers.remove(fold)
        outer_validation = (fold + 1) % n_folds # Slightly different between implementations!!!

        print(f"fold: {fold}, outer_validation: {outer_validation}")

        # Select holdout and training data based on fold column == or != to fold
        holdout_df = df[df["cv_fold"] == fold]
        train_df = df[df["cv_fold"] != fold]

        train_x = torch.tensor(train_df.drop(["y1", "cv_fold"], axis=1).values, dtype=torch.float32)
        train_y = torch.tensor(train_df["y1"].values, dtype=torch.float32).unsqueeze(1)
        training_data = TensorDataset(train_x, train_y)

        holdout_x = torch.tensor(holdout_df.drop(["y1", "cv_fold"], axis=1).values, dtype=torch.float32)
        holdout_y = torch.tensor(holdout_df["y1"].values, dtype=torch.float32).unsqueeze(1)
        holdout_data = TensorDataset(holdout_x, holdout_y)

        # Temp fixed hyper params:
        nodes_layer_one = 50
        nodes_layer_two = 50
        activation_f_one = nn.Softplus()
        activation_f_two = nn.Softplus()
        # loss_f = nn.BCEWithLogitsLoss() ----- Does not seem to have solid alternatives
        learning_rate = 0.01
        batch_size = int(0.2 * training_data.tensors[0].shape[0])
        stopping_criterion = 500
        regularization_lambda = 0.1

        model = trained_neural_net(training_data=training_data,
                                   nodes_layer_one=nodes_layer_one,
                                   activation_f_one=activation_f_one,
                                   nodes_layer_two=nodes_layer_two,
                                   activation_f_two=activation_f_two,
                                   learning_rate=learning_rate,
                                   batch_size=batch_size,
                                   stopping_criterion=stopping_criterion,
                                   regularization_lambda=regularization_lambda)

        # Printing in sample and oos pred quality
        loss_f = nn.BCEWithLogitsLoss()
        train_losses = []
        with torch.no_grad():
            for batch_X, batch_y in training_data:
                preds = model(batch_X)
                loss = loss_f(preds, batch_y)
                train_losses.append(loss.item())

        print(f"In-sample Loss: {sum(train_losses) / len(train_losses):.4f}")

        val_losses = []
        with torch.no_grad():
            for batch_X, batch_y in holdout_data:
                preds = model(batch_X)
                loss = loss_f(preds, batch_y)
                val_losses.append(loss.item())
        print(f"Oos Loss: {sum(val_losses) / len(val_losses):.4f}")

        in_sample_losses.extend(train_losses)
        oos_losses.extend(val_losses)

    print()
    print("Average over all folds:")
    print(f"In-sample loss: {sum(in_sample_losses) / len(in_sample_losses):.4f}")
    print(f"Oos loss: {sum(oos_losses) / len(oos_losses):.4f}")


# Two-layered feed-forward Neural Net class
class BinaryNeuralNet(nn.Module):
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
            nn.Linear(in_features= nodes_layer_two, out_features= 1))

    def forward(self, x):
        return self.net(x)

# Creates and trains the neural net
#
# Hyperparameters to be tuned / to be chosen:
# xFirst layer:
# x  # of nodes
# x  Activation function
# xSecond layer:
# x  # of nodes
# x  Activation function
# Starting values for beta
# Loss function (Kinda needs to be BCE logits smth smth loss criterion according to gpt)
# xLearning rate
# xMini-batch subgradient descent: batch size
# ?Stopping criterion (currently beta update iteration count)
# xRegularization parameter \lambda (currently lambda for ridge, gpt recommended, also recommended dropout)
def trained_neural_net(training_data: TensorDataset, nodes_layer_one: int, nodes_layer_two: int,
                       activation_f_one: nn.Module, activation_f_two: nn.Module, learning_rate: float,
                       batch_size: int, stopping_criterion: int, regularization_lambda: float):
    feature_count = training_data.tensors[0].shape[1]
    training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

    model = BinaryNeuralNet(feature_count,
                            nodes_layer_one, activation_f_one,
                            nodes_layer_two, activation_f_two)

    optimizer = optim.Adam(model.parameters(),
                           lr=learning_rate,
                           weight_decay=regularization_lambda)

    loss_f = nn.BCEWithLogitsLoss()

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

if __name__ == "__main__":
    main()