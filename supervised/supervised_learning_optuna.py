import os
import gymnasium as gym
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Subset
import random
import torch.nn.functional as F
import pickle
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import numpy as np
import optuna
import torch.nn as nn
import time
import argparse

################################## set device ##################################
print("============================================================================================")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print("============================================================================================")


def parse_arguments():
    # Argument parser
    parser = argparse.ArgumentParser(description='Supervised Learning with Optuna')
    parser.add_argument('--pkl_file_path', type=str, required=True, help='Base path to pkl file')
    parser.add_argument('--base_output_path', type=str, required=True, help='Base path to save the output file')
    parser.add_argument('--n_trials', type=int, required=True, default=50,
                        help='Number of trials for Optuna optimization')
    parser.add_argument("--observation", type=str, default="Kinematics",
                        choices=["Kinematics", "OccupancyGrid"],
                        help="Defines the observation type of the highway environment.")
    parser.add_argument('--seed', type=int, default=5, help="Random seed.")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs to train.")

    return parser.parse_args()


def set_random_seed(seed=0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class TrajectoryDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        _, _, state, action, _, discounted_cost = self.data[idx]

        # Convert state/action to torch tensors if needed
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float)
        if not torch.is_tensor(action):
            # If action is a scalar (int), convert it to float tensor
            if isinstance(action, int):
                action = torch.tensor([float(action)], dtype=torch.float)
            else:
                action = torch.tensor(action, dtype=torch.float)

        # Ensure tensors are at least 1-dimensional
        if state.dim() == 0:
            state = state.unsqueeze(0)
        if action.dim() == 0:
            action = action.unsqueeze(0)

        # Concatenate state and action
        input_features = torch.cat((state.view(-1), action.view(-1)))

        # discounted_cost is already continuous, use as-is
        # label = torch.tensor(discounted_cost, dtype=torch.float).view(-1, 1)  # shape [1]
        label = torch.tensor(discounted_cost, dtype=torch.float).unsqueeze(0)  # shape [1]

        return input_features, label


class Shield(nn.Module):
    def __init__(self, input_size, num_layers, loss_fn, lr, hidden_dim, activation):
        super(Shield, self).__init__()
        layers = [nn.Linear(input_size, hidden_dim)]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation)
        layers.append(nn.Linear(hidden_dim, 1))
        # add sigmoid as last layer
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

        self.criterion = loss_fn
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        return self.model(x)

    def save_model(self, save_path, trial_num):
        save_path = save_path + f'/shield_model_trial_{trial_num}.pth'
        torch.save(self.state_dict(), save_path)

    def load_model(self, load_path, trial_num):
        load_path = load_path + f'/shield_model_trial_{trial_num}.pth'
        self.load_state_dict(torch.load(load_path))


def create_batches(dataset, batch_size):
    # data_list = list(data_list)
    # Shuffle the data list randomly
    data_list = [(dataset[i][0], dataset[i][1]) for i in range(len(dataset))]
    random.shuffle(data_list)

    batches = []
    for i in range(0, len(data_list), batch_size):
        batch = data_list[i:i + batch_size]
        batches.append(batch)

    return batches


def get_input_size(state, action):
    # Ensure these are tensors
    if not torch.is_tensor(state):
        t_state = torch.tensor(state, dtype=torch.float)
    if isinstance(action, int):
        t_action = torch.tensor([float(action)], dtype=torch.float)
    else:
        t_action = torch.tensor(action, dtype=torch.float)

    input_size = len(t_state) + len(t_action)
    return input_size


def objective(trial, epochs, input_size, train_loader, test_loader, trials_results, output_path):
    start_time = time.time()

    lr_shield = trial.suggest_float('lr_shield', 1e-5, 1e-1, log=True)  # Extended range: from 0.00001 to 0.1
    num_layers = trial.suggest_int('num_layers', 1, 4)  # Extended range: from 1 to 4
    hidden_dim = 2 ** trial.suggest_int('hidden_dim_exponent', 5, 9)  # Extended range: from 32 to 512
    activation_name = trial.suggest_categorical('activation', ['relu', 'tanh'])

    if activation_name == 'relu':
        activation = nn.ReLU()
    else:
        activation = nn.Tanh()

    loss_fun = nn.MSELoss()  # change to BCE if predicting non-continuous value
    shield_model = Shield(input_size=input_size, num_layers=num_layers, loss_fn=loss_fun,
                          lr=lr_shield, hidden_dim=hidden_dim, activation=activation).to(device)

    print(f"Trial {trial.number}: lr_shield={lr_shield}, train_batch_size={train_loader.batch_size}, "
          f"num_layers={num_layers}, hidden_dim={hidden_dim}, activation={activation_name}")

    # shield_model.load_model(output_path)

    shield_model.train()
    for epoch in range(epochs):
        epoch_losses = []
        for inputs, targets in train_loader:
            if inputs is None or targets is None:
                print("None")
            inputs = inputs.float().to(device)
            targets = targets.float().to(device)

            shield_model.optimizer.zero_grad()
            outputs = shield_model(inputs)
            loss = shield_model.criterion(outputs, targets)
            loss.backward()
            shield_model.optimizer.step()

            epoch_losses.append(loss.item())
        avg_epoch_loss = np.mean(epoch_losses)
        print(f"Trial {trial.number}, Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss:.4f}")

    # shield_model.save_model(output_path, trial.number)

    # Evaluation on test set
    shield_model.eval()
    test_losses = []
    costs_predictions = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.float().to(device)
            targets = targets.float().to(device)
            outputs = shield_model(inputs)
            loss = shield_model.criterion(outputs, targets)
            test_losses.append(loss.item())
            costs_predictions.extend(zip(targets.cpu().numpy().flatten(), outputs.cpu().numpy().flatten()))

    average_mse = np.mean(test_losses)

    # Save predictions for analysis
    df = pd.DataFrame(costs_predictions, columns=['cost', 'prediction'])
    df.to_csv(os.path.join(output_path, f"costs_predictions_trial_{trial.number}.csv"), index=False)

    end_time = time.time()
    trial_duration = end_time - start_time
    print(f"Trial {trial.number} completed in {trial_duration:.2f} seconds with MSE={average_mse}.")

    trials_results.append({
        'trial_number': trial.number,
        'lr_shield': lr_shield,
        'train_batch_size': train_loader.batch_size,
        'num_layers': num_layers,
        'hidden_dim': hidden_dim,
        'activation': activation_name,
        'average_mse': average_mse
    })
    print(f"Finished trial {trial.number} average_mse={average_mse}")

    return average_mse


def main():
    args = parse_arguments()
    set_random_seed(args.seed)

    start_time = int(time.time())
    folder_path = os.path.join(args.base_output_path, f"{args.observation}_{start_time}")
    os.makedirs(folder_path, exist_ok=True)

    with open(args.pkl_file_path, 'rb') as file:
        all_samples = pickle.load(file)
        print(f"Loaded {len(all_samples)} samples from {args.pkl_file_path}.")

    # Split the merged dataset into train and test datasets
    train_size = int(0.8 * len(all_samples))
    test_size = len(all_samples) - train_size
    dataset = TrajectoryDataset(all_samples)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Determine input size
    s, _ = train_dataset[0]
    input_size = s.numel()

    trials_results = []  # List to store trial results
    study = optuna.create_study(direction='minimize')
    study.optimize(
        lambda trial: objective(
            trial=trial,
            epochs=args.epochs,
            input_size=input_size,
            train_loader=DataLoader(train_dataset, batch_size=2 ** trial.suggest_int('train_batch_exponent', 5, 10),
                                    shuffle=True),
            test_loader=DataLoader(test_dataset, batch_size=256, shuffle=False),
            trials_results=trials_results,
            output_path=folder_path
        ),
        n_trials=args.n_trials
    )

    print('Best hyperparameters: ', study.best_params)
    print('Best score (MSE): ', study.best_value)

    # Save trial results to file
    save_output_path = os.path.join(folder_path, f"{args.n_trials}_trials.csv")
    results_df = pd.DataFrame(trials_results)
    results_df.to_csv(save_output_path, index=False)
    print("Results saved to:", save_output_path)


if __name__ == '__main__':
    main()

# Determine input size
# _, _, example_state, example_action, _, _ = all_samples[0]
# input_size = get_input_size(example_state, example_action)
# # Evaluation on test set
# shield_model.eval()
# test_losses = []
# with torch.no_grad():
#     for inputs, targets in test_loader:
#         inputs = inputs.float().to(device)
#         targets = targets.float().to(device)
#         outputs = shield_model(inputs)
#         loss = shield_model.criterion(outputs, targets)
#         test_losses.append(loss.item())
# average_mse = np.mean(test_losses)
#
# # Save predictions for analysis
# shield_model.eval()
# costs_predictions = []
# with torch.no_grad():
#     for inputs, targets in test_loader:
#         inputs = inputs.float().to(device)
#         outputs = shield_model(inputs)
#         costs_predictions.extend(zip(targets.cpu().numpy().flatten(), outputs.cpu().numpy().flatten()))


# # Evaluate on train dataset (for simplicity)
# shield_model.eval()
# with torch.no_grad():
#     train_inputs = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))]).to(device)
#     train_targets = torch.stack([train_dataset[i][1] for i in range(len(train_dataset))]).to(device)
#     train_outputs = shield_model(train_inputs)
#     train_loss = shield_model.criterion(train_outputs, train_targets)
# average_mse = train_loss.item()
#
# # Save predictions for analysis
# costs_predictions = list(zip(train_targets.cpu().numpy().flatten(), train_outputs.cpu().numpy().flatten()))
