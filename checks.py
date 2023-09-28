"""
import os
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#files = [torch.load("models/Test_add_frames/from_GPU/stats_k=1.log"), torch.load("models/Test_add_frames/from_GPU/stats_k=3.log"), torch.load("models/Test_add_frames/from_GPU/stats_k=5.log")]
files = [torch.load("models/ShieldPPO_1_20230912-223249/stats.log")]


for i, file in enumerate(files):
    print(file[0])
"""


import os
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#architecture comparing



#files = [torch.load("models/Test_add_frames/from_GPU/stats_k=1.log"), torch.load("models/Test_add_frames/from_GPU/stats_k=2.log"), torch.load("models/Test_add_frames/from_GPU/stats_k=3.log"), torch.load("models/Test_add_frames/from_GPU/stats_k=5.log")]
#files = [torch.load("models/Test_MaskingTres/from_GPU/stats_masking=0K.log"), torch.load("models/Test_MaskingTres/from_GPU/stats_masking=10K.log"), torch.load("models/Test_MaskingTres/from_GPU/stats_masking=50K.log"), torch.load("models/Test_MaskingTres/from_GPU/stats_masking=100K.log"), torch.load("models/Test_MaskingTres/from_GPU/stats_masking=150K.log"), torch.load("models/Test_MaskingTres/from_GPU/stats_masking=300K.log")]


files = [torch.load("models/ShieldPPO_1_20230912-223249/stats.log"), torch.load("models/no_lstm_stats.log")]

#path_to_save = "models/Test_add_frames/from_GPU/plots.png"
#path_to_save = "models/Test_MaskingTres/from_GPU/plots.png"
path_to_save = "models/architectures_comparison_actor_critic.png"


#path_to_save_df = "models/Test_add_frames/from_GPU/comparison_table.csv"
#path_to_save_df = "models/Test_MaskingTres/from_GPU/comparison_table.csv"
path_to_save_df = "models/architectures_comparison_table_actor_critic.csv"


scores_for_table = {}

#keys = ["k_last_states=1", "k_last_states=2", "k_last_states=3", "k_last_states=5"]
#keys = ["masking_tresh = 0", "k_last_states=10K", "k_last_states=50K", "k_last_states=100K", "masking_tresh = 150K", "masking_tresh = 300K"]
keys = ["LSTM - Actor Critic & Shield ", "NO LSTM - FROM THE ORIGINAL PAPER"]

for i, file in enumerate(files):
    rewards, costs = file[1][:101000], file[2][:101000]
    print(rewards)
    avg_rewards = round(sum(rewards) / len(rewards), 5)
    avg_costs = round(sum(costs) / len(costs), 5)
    scores_for_table[keys[i]] = (avg_rewards, avg_costs)


# Create a DataFrame from the dictionary
df = pd.DataFrame(scores_for_table.items(), columns=['Parameter Value', 'Average Rewards and Costs'])

# Split the tuple into two columns
df[['Average Rewards', 'Average Costs']] = pd.DataFrame(df['Average Rewards and Costs'].tolist(), index=df.index)

# Drop the original combined column
df = df.drop(columns=['Average Rewards and Costs'])

df.to_csv(path_to_save_df, index=False)


def cost_to_cost_rate(array):
    global acc, i, tmp
    acc = 0
    for i in range(len(array[2])):
        tmp = array[2][i]
        array[2][i] += acc
        array[2][i] /= (i+1)
        acc += tmp

for file in files:
    cost_to_cost_rate(file)

"""
def smooth(x):
    # calculates the mean over a window of 50 data points (timestamps)
    n = x.shape[0]
    b = np.zeros((n,))
    for i in range(n):
        b[i] = x[max(0, i-50):min(n, i+50)].mean()
    return b

"""


def smooth(x):
    return x

# Create a figure with 6 subplots (2 rows and 3 columns)
fig, axs = plt.subplots(2, len(files), figsize=(30, 16))

# Define colors for each experiment
colors = ['r', 'b', 'g', 'purple', 'orange', 'cyan']

# Define labels for the experiments
#experiment_labels = ['last_states = 1 ', 'last_states = 2 ', 'last_states = 3 ', 'last_states = 5 ']


#experiment_labels = ["masking = 0", "masking = 10K", "masking = 50K", "masking = 100K", "masking = 150K", "masking = 300K"]


#Loop through rows (experiments)
# Loop through rows (experiments)
for i, data in enumerate(files):
    # Plot return in the first column
    axs[0, i].set_title(f'Return - {keys[i]}')
    axs[0, i].set_xlabel("Steps")
    axs[0, i].set_ylabel("Return")
    # Set x-axis limits to display only the first 100,000 points
    axs[0, i].set_xlim(0, 100000)
    axs[0, i].plot(data[0][1:100001], np.array(data[1][1:100001]), label=keys[i], color=colors[i])

    # Plot mistake rate in the second column
    axs[1, i].set_title(f'Mistake Rate - {keys[i]}')
    axs[1, i].set_xlabel("Steps")
    axs[1, i].set_ylabel("Mistake Rate")

    # Set x-axis limits to display only the first 100,000 points
    axs[1, i].set_xlim(0, 100000)

    axs[1, i].plot(data[0][1:100001], np.array(data[2][1:100001]), label=keys[i], color=colors[i])

# Adjust subplot layout
plt.tight_layout()

# Save the figure
plt.savefig(path_to_save, dpi=100)






