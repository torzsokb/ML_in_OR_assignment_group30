import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Diff with regression:
# Second head to NN
# Weighing of loss functions for Reg vs Class

def main():
    df = pd.read_csv(r"C:\Users\roela\PycharmProjects\ML_in_OR_assignment_group30\documents\data\processed_data.csv")
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

        train_x = torch.tensor(train_df.drop(["y1", "y2", "cv_fold"], axis=1).values, dtype=torch.float32)
        train_y_class = torch.tensor(train_df["y1"].values, dtype=torch.float32).unsqueeze(1)
        train_y_reg = torch.tensor(train_df["y2"].values, dtype=torch.float32).unsqueeze(1)
        training_data = TensorDataset(train_x, train_y_class, train_y_reg)

        holdout_x = torch.tensor(holdout_df.drop(["y1", "y2", "cv_fold"], axis=1).values, dtype=torch.float32)
        holdout_y_class = torch.tensor(holdout_df["y1"].values, dtype=torch.float32).unsqueeze(1)
        holdout_y_reg = torch.tensor(holdout_df["y2"].values, dtype=torch.float32).unsqueeze(1)
        holdout_data = TensorDataset(holdout_x, holdout_y_class, holdout_y_reg)

        # Temp fixed hyper params:
        nodes_layer_one = 10
        nodes_layer_two = 10
        activation_f_one = nn.Softplus()
        activation_f_two = nn.Softplus()
        loss_f_reg = nn.MSELoss()
        learning_rate = 0.01
        batch_size = int(0.2 * training_data.tensors[0].shape[0])
        stopping_criterion = 250
        regularization_lambda = 0.1

        model = trained_neural_net(training_data=training_data,
                                   nodes_layer_one=nodes_layer_one,
                                   activation_f_one=activation_f_one,
                                   nodes_layer_two=nodes_layer_two,
                                   activation_f_two=activation_f_two,
                                   loss_f=loss_f_reg,
                                   learning_rate=learning_rate,
                                   batch_size=batch_size,
                                   stopping_criterion=stopping_criterion,
                                   regularization_lambda=regularization_lambda)

        # Printing in sample and oos pred quality
        model.eval()
        loss_f_class = nn.BCEWithLogitsLoss()
        train_losses = []
        with torch.no_grad():
            for batch_X, batch_y_class, batch_y_reg in training_data:
                preds_class, preds_reg = model(batch_X)

                reg_loss = loss_f_reg(preds_reg, batch_y_reg)
                class_loss = loss_f_class(preds_class, batch_y_class)

                # TODO: consider weights (or standardize reg data) KEEP SAME AS IMPLEMENTATION further down the file!!!
                loss = reg_loss + class_loss

                train_losses.append(loss.item())

        print(f"In-sample Loss: {sum(train_losses) / len(train_losses):.4f}")

        val_losses = []
        with torch.no_grad():
            for batch_X, batch_y in holdout_data:
                preds = model(batch_X)
                loss = loss_f_reg(preds, batch_y)
                val_losses.append(loss.item())
        print(f"Oos Loss: {sum(val_losses) / len(val_losses):.4f}")

        in_sample_losses.extend(train_losses)
        oos_losses.extend(val_losses)

    print()
    print("Average over all folds:")
    print(f"In-sample loss: {sum(in_sample_losses) / len(in_sample_losses):.4f}")
    print(f"Oos loss: {sum(oos_losses) / len(oos_losses):.4f}")


# Two-layered feed-forward Neural Net class
class CombinedNeuralNet(nn.Module):
    def __init__(self,
                 feature_count: int,
                 nodes_layer_one: int,
                 activation_f_one: nn.Module,
                 nodes_layer_two: int,
                 activation_f_two: nn.Module):
        super().__init__() # super(RegressionNeuralNet, self).etc by GPT
        # self.flatten = nn.Flatten() # part of pytorch tutorial, not of gpt
        self.shared_net = nn.Sequential(
            nn.Linear(in_features= feature_count, out_features= nodes_layer_one),
            activation_f_one,
            nn.Linear(in_features= nodes_layer_one, out_features= nodes_layer_two),
            activation_f_two)

        self.class_head = nn.Linear(in_features= nodes_layer_two, out_features= 1)
        self.reg_head = nn.Linear(in_features= nodes_layer_two, out_features= 1)

    def forward(self, x):
        h = self.shared_net(x)
        class_out = self.class_head(h)
        reg_out = self.reg_head(h)

        return class_out, reg_out

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
# xLoss function (i.e.: nn.MSELoss() )
# xLearning rate
# xMini-batch subgradient descent: batch size
# ?Stopping criterion (currently beta update iteration count, also need to consider stopping crit hyper params)
# xRegularization parameter \lambda (currently lambda for ridge, gpt recommended, also recommended dropout)
def trained_neural_net(training_data: TensorDataset, nodes_layer_one: int, nodes_layer_two: int,
                       activation_f_one: nn.Module, activation_f_two: nn.Module, loss_f: nn.Module, learning_rate: float,
                       batch_size: int, stopping_criterion: int, regularization_lambda: float):
    feature_count = training_data.tensors[0].shape[1]
    training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

    model = CombinedNeuralNet(feature_count,
                              nodes_layer_one, activation_f_one,
                              nodes_layer_two, activation_f_two)

    optimizer = optim.Adam(model.parameters(),
                           lr=learning_rate,
                           weight_decay=regularization_lambda)

    class_loss_criterion = nn.BCEWithLogitsLoss()
    reg_loss_criterion = loss_f

    for beta_iter in range(stopping_criterion):
        for batch_X, batch_y_class, batch_y_reg in training_loader:
            # Forward pass
            class_pred, reg_pred= model(batch_X)
            class_loss = class_loss_criterion(class_pred, batch_y_class)
            reg_loss = reg_loss_criterion(reg_pred, batch_y_reg)

            # TODO: consider weights (or standardize reg data) KEEP SAME AS IMPLEMENTATION above in the file!!!
            loss = reg_loss + class_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model

if __name__ == "__main__":
    main()