import torch

def forward(self, s, last_informative_layers, a):
    """
    tensor(8,3,71)
    tensor(8,)
tensor(8,)

    s - last k_states (or less) for each sample in the batch
    last_informative_layers - index of the last informative layer - for each sample in the batch
    a - tensor of actions - for each sample in the batch
    """
    a = self.encode_action(a)
    lstm_output, _ = self.lstm(s)
    # PREVIOUS - Selecting the last layer - hidden state which captures the relevant information from entire sequence (states)
    # lstm_output_last = lstm_output[:, -1, :]  # Shape: [1, hidden_dim]
    # MeetingComment - (!) select the last informative layer
    # MeetingComment - ASK SHAHAF: DOES IT RETURN THE SCORE FOR A SPECEIFIC ACTION? SEE THE DIFFERENT BETWEEN 'SELECT_ACTION' AND 'UPDATE-SHIELD'
    last_informative_layers_casted = [int(ind.item()) for ind in last_informative_layers]
    # lstm_output_last = torch.stack([t[int(last_informative_layers[i].item())] for i, t in enumerate(lstm_output)])
    first_dim_len = len(lstm_output) - 1
    lstm_output_last = lstm_output[first_dim_len, last_informative_layers_casted, :]
    x = torch.cat([lstm_output_last, a], -1)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    safety_scores = torch.sigmoid(self.fc3(x)).squeeze(-1)
    return safety_scores.squeeze(-1)