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
# Print the device being used
print(f"Using device: {device}")

print("============================================================================================")

# Argument parser
parser = argparse.ArgumentParser(description='Supervised Learning with Optuna')
parser.add_argument('--pkl_file_path', type=str, required=True, help='Base path to pkl file')
parser.add_argument('--base_output_path', type=str, required=True, help='Base path to save the output file')
parser.add_argument('--n_trials', type=int, required=True, default=50, help='Number of trials for Optuna optimization')
parser.add_argument("--observation", type=str, default="Kinematics",
                    choices=["Kinematics", "OccupancyGrid"],
                    help="Defines the observation type of the highway environment.")

args = parser.parse_args()
start_time = int(time.time())
save_output_path = os.path.join(args.base_output_path, f"{args.n_trials}_trials_{start_time}.csv")
os.makedirs(os.path.dirname(save_output_path), exist_ok=True)

random_seed = 0

torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

with open(args.pkl_file_path, 'rb') as file:
    all_samples = pickle.load(file)
    print(f"Successfully loaded 'samples_path' with {len(all_samples)} samples.")


class TrajectoryDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state = self.data[idx][0]  # State vector (tensor)
        action = self.data[idx][1]  # Action (tensor)
        cost = self.data[idx][2]  # Cost (tensor)
        # uncomment for masking
        cost = (cost >= 1).float()
        input_features = torch.cat((state.view(-1), action.float()))  # Concat state and action
        return input_features, cost.float()


# Split the merged dataset into training and testing datasets
train_size = int(0.8 * len(all_samples))
test_size = len(all_samples) - train_size
train_dataset, test_dataset = random_split(TrajectoryDataset(all_samples), [train_size, test_size])


class Shield(nn.Module):
    def __init__(self, input_size, num_layers, hidden_dim, activation):
        super(Shield, self).__init__()
        layers = [nn.Linear(input_size, hidden_dim)]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation)
        layers.append(nn.Linear(hidden_dim, 1))
        # add sigmoid as last layer
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def save_model(self, save_path):
        save_path = save_path + '/shield_model.pth'
        torch.save(self.state_dict(), save_path)

    def load_model(self, load_path):
        load_path = load_path + '/shield_model.pth'
        self.load_state_dict(torch.load(load_path))


def create_batches(data_list, batch_size):
    # Shuffle the data list randomly
    data_list = list(data_list)
    random.shuffle(data_list)

    # Create a list of batches
    batches = []

    # Iterate over the data in steps of batch_size
    for i in range(0, len(data_list), batch_size):
        # Slice the data from the current index i to i + batch_size
        batch = data_list[i:i + batch_size]
        # Append the batch to the batches list
        batches.append(batch)

    return batches


if args.highway_observation_type == "OccupancyGrid":
    input_size = 257
if args.highway_observation_type == "Kinematics":
    input_size = 26
if args.highway_observation_type == "TimeToCollision":
    input_size = 91


def objective(trial):
    start_time = time.time()
    lr_shield = trial.suggest_loguniform('lr_shield', 1e-5, 1e-1)  # Extended range: from 0.00001 to 0.1
    batch_size = trial.suggest_int('batch_size', 100, 10000)  # Extended range: from 100 to 10,000
    # Convert mini_batch_size to a range instead of categorical for broader exploration
    mini_batch_size = trial.suggest_int('mini_batch_size', 16, 2000)  # Extended range: from 16 to 2000
    num_layers = trial.suggest_int('num_layers', 1, 10)  # Extended range: from 1 to 10
    hidden_dim = trial.suggest_int('hidden_dim', 16, 512)  # Extended range: from 16 to 512

    activation_name = trial.suggest_categorical('activation', ['relu', 'tanh'])
    # print(f"Starting trial {trial.number} with params: lr_shield={lr_shield}, batch_size={batch_size}, mini_batch_size={mini_batch_size}, num_layers={num_layers}, hidden_dim={hidden_dim}, activation={activation_name}")

    if activation_name == 'relu':
        activation = nn.ReLU()
    else:
        activation = nn.Tanh()

    # determine input size

    shield_model = Shield(input_size=input_size, num_layers=num_layers, hidden_dim=hidden_dim,
                          activation=activation).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(shield_model.parameters(), lr=lr_shield)
    print("Start to create train and test loaders")

    batches = create_batches(train_dataset, batch_size)
    train_loader = DataLoader(train_dataset, batch_size=mini_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=mini_batch_size, shuffle=False)
    print("Finished creating train and test loaders")
    # shield_model.load_model(args.base_output_path)
    for epoch in range(5):
        for train_batch in batches:
            shield_model.train()
            total_loss = 0
            train_loader = DataLoader(train_batch, batch_size=mini_batch_size, shuffle=True)
            for inputs, targets in train_loader:
                inputs = inputs.float().to(device)
                targets = targets.float().view(-1, 1).to(device)
                optimizer.zero_grad()
                outputs = shield_model(inputs)
                loss = criterion(outputs, targets)
                print("Trial: ", trial.number, " Loss: ", loss)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
    # shield_model.save_model(args.base_output_path)

    shield_model.eval()
    total_mse = 0
    total_samples = 0
    costs_predictions = []

    inputs = torch.stack([t[0] for t in train_dataset])
    targets = torch.stack([t[1] for t in train_dataset])
    with torch.no_grad():
        inputs = inputs.float().to(device)
        targets = targets.float().view(-1, 1).to(device)
        outputs = shield_model(inputs)
        loss = criterion(outputs, targets)
        print("Trial: ", trial.number, "Train loss after training: ", loss)
        total_mse += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)
        costs_predictions.extend(zip(targets.cpu().numpy(), outputs.cpu().numpy()))

    costs_predictions = [(cost[0], prediction[0]) for cost, prediction in costs_predictions]

    average_mse = total_mse / total_samples
    df = pd.DataFrame(costs_predictions, columns=['cost', 'prediction'])
    df.to_csv(args.base_output_path + f'/costs_predictions_trial_{trial.number}.csv', index=False)

    end_time = time.time()
    trial_duration = end_time - start_time
    print(f"Trial {trial.number} took {trial_duration:.2f} seconds.")
    trial_results.append({
        'trial_number': trial.number,
        'lr_shield': lr_shield,
        'batch_size': batch_size,
        'mini_batch_size': mini_batch_size,
        # 'num_layers': num_layers,
        # 'hidden_dim': hidden_dim,
        'activation': activation_name,
        'average_mse': average_mse
    })
    #    print(f"Finished trial {trial.number} results: lr_shield={lr_shield}, batch_size={batch_size}, mini_batch_size={mini_batch_size}, num_layers={num_layers}, hidden_dim={hidden_dim}, activation={activation_name}, average_mse={average_mse}")

    return average_mse


# List to store trial results

trial_results = []

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=args.n_trials)

print('Best hyperparameters: ', study.best_params)
print('Best score for relevant hyperpamaters: ', study.best_value)

# Save trial results to file
results_df = pd.DataFrame(trial_results)
results_df.to_csv(save_output_path, index=False)
