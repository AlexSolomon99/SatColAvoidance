import os
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import datetime
import torch

import utils
import models
import dataprocessing

from policy_methods_utils import PolicyMethodsUtils

import sys

sys.path.append(r'E:\Alex\UniBuc\MasterThesis\gym-satellite-ca')

from gym_satellite_ca.envs import satDataClass

# set device
device = torch.device('cuda')

# constant paths
BASE = "./"
DATA_PATH = os.path.join(BASE, "data")

# paths
sat_data_config = os.path.join(DATA_PATH, "default_sat_data_config.json")

# setting up the satellite data and init config of the environment
init_sat = utils.get_sat_data_env(sat_data_config)

# setting up the environment
env = gym.make('gym_satellite_ca:gym_satellite_ca/CollisionAvoidance-v0',
               satellite=init_sat)

# set up the observation processing class
tca_time_lapse_max_abs_val = env.observation_space['tca_time_lapse'].high[0]
data_preprocessing = dataprocessing.data_processing.ObservationProcessing(satellite_data=env.unwrapped.satellite,
                                                                          tca_time_lapse_max_abs_val=tca_time_lapse_max_abs_val)

# set up neural net configuration
nn_conf = {
    "init_layer": 4,
    "hidden_layer": 16,
    "output_layer": 6
}

# instantiate the policy and its utilities
nn_policy = models.NN_NeuralNet.SatColAvoidPolicy(conf=nn_conf)
policy_utils = PolicyMethodsUtils(observation_processing=data_preprocessing, device=device)
optimizer = policy_utils.instantiate_loss_fnc_optimiser(policy=nn_policy)

# set up training variables
train_eval_steps = 100
train_total_losses = torch.tensor([], device=device)
train_rewards_sum_list = torch.tensor([], device=device)
eval_rewards_sum_list = torch.tensor([], device=device)

print(f"{datetime.datetime.now()} - Started training")
for steps in range(train_eval_steps):
    # train the model for n times
    train_losses, train_arr_raw_rewards = policy_utils.train_policy(policy=nn_policy, game_env=env,
                                                                    optimizer=optimizer, num_train_iterations=5)

    # eval the model for n times
    eval_arr_raw_rewards = policy_utils.eval_policy(policy=nn_policy, game_env=env,
                                                    optimizer=optimizer, num_eval_iterations=1)

    train_rewards_sum = torch.sum(train_arr_raw_rewards.mean(axis=1))
    eval_rewards_sum = torch.sum(eval_arr_raw_rewards.mean(axis=1))

    print(f"{datetime.datetime.now()} - Epoch {steps} - Train Reward MeanSum: {train_rewards_sum} "
          f"| Eval Reward MeanSum: {eval_rewards_sum}")

    train_total_losses = torch.cat((train_total_losses, train_losses))
    train_rewards_sum_list = torch.cat((train_rewards_sum_list, train_rewards_sum))
    eval_rewards_sum_list = torch.cat((eval_rewards_sum_list, eval_rewards_sum))

n_loss = np.arange(0, len(train_total_losses))
n_rewards = np.arange(0, len(train_rewards_sum_list))

plt.plot(n_loss, train_total_losses)
plt.title(f"Losses")
plt.xlabel(f"Games played")
plt.ylabel(f"Loss value")
plt.grid()
plt.show()

plt.plot(n_rewards, train_rewards_sum_list)
plt.plot(n_rewards, eval_rewards_sum_list)
plt.title(f"Rewards received")
plt.xlabel(f"Games played")
plt.ylabel(f"Rewards received")
plt.grid()
plt.show()
