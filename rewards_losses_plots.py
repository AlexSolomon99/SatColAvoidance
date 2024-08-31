import os
import numpy as np
import matplotlib.pyplot as plt
import utils

data_dir_path = r"E:\Alex\UniBuc\MasterThesis\results\cartesian_coordinates\discrete_action\dqn\dqn_model_4_dir_as_train_thrust"
file_name = "Losses_Reward_dict.json"
file_path = os.path.join(data_dir_path, file_name)

json_data = utils.read_json(file_path)
losses = json_data["Losses"]
rewards = json_data["Rewards"]


rewards_plot, ax = plt.subplots(1, 1, figsize=(15, 6))
x_axis_data = np.arange(len(rewards))
x_label = 'Train Epoch'

ax.plot(x_axis_data, rewards, color='navy')

# Add labels and titles
ax.set_xlabel(x_label)
ax.set_ylabel('Reward Value', fontsize=22)
ax.grid(True)
ax.tick_params(axis='both', labelsize=18)
ax.set_xlabel(x_label, fontsize=22)

rewards_plot.savefig(os.path.join(data_dir_path, f"RewardsPlotTrain.png"))
plt.close(rewards_plot)

# compute the keplerian elements plots
rewards_plot, axs = plt.subplots(2, 1, figsize=(25, 20))
x_axis_data = np.arange(len(rewards))
x_label = 'Train Epoch'

# plot the data
axs[0].plot(x_axis_data, rewards, color='navy')
axs[1].plot(x_axis_data, losses, color='darkred')

# Add labels and titles
label_pad = 24
ylabel_fontsize = 22
axs[0].set_ylabel('Rewards', fontsize=ylabel_fontsize, labelpad=label_pad)
axs[1].set_ylabel('Losses', fontsize=ylabel_fontsize, labelpad=label_pad)

# keplerian_plot.suptitle('Keplerian Elements Differences over Time', fontsize=36)
for ax_idx in range(2):
    if ax_idx == 1:
        axs[ax_idx].set_xlabel(x_label, fontsize=22)
    axs[ax_idx].grid(True)
    axs[ax_idx].tick_params(axis='both', labelsize=18)

rewards_plot.savefig(os.path.join(data_dir_path, f"RewardsLossesPlotTrain.png"))
plt.close(rewards_plot)
