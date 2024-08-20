import copy
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

# model name
MODEL_NAME = "policy_model"

# constant paths
BASE = r"E:\Alex\UniBuc\MasterThesis\src"
DATA_PATH = os.path.join(BASE, "data")
RECORDED_MODELS_PATH = os.path.join(BASE, "recorded_models")
POLICY_METHODS_MODELS = os.path.join(RECORDED_MODELS_PATH, "policy_methods_models")

if not os.path.isdir(RECORDED_MODELS_PATH):
    os.mkdir(RECORDED_MODELS_PATH)

if not os.path.isdir(POLICY_METHODS_MODELS):
    os.mkdir(POLICY_METHODS_MODELS)

# paths - set up the model record dictionary
sat_data_config = os.path.join(DATA_PATH, "default_sat_data_config.json")
policy_methods_record_dict_path = os.path.join(POLICY_METHODS_MODELS, "policy_methods_record.json")

if not os.path.isfile(policy_methods_record_dict_path):
    model_record_dict = {}
    model_record_last_idx = 0
else:
    model_record_dict = utils.read_json(json_path=policy_methods_record_dict_path)
    model_record_last_idx = len(model_record_dict.keys())
best_model_dir_path = os.path.join(POLICY_METHODS_MODELS, f"{MODEL_NAME}_{model_record_last_idx + 1}_dir")
best_model_path = os.path.join(best_model_dir_path, f"{MODEL_NAME}_{model_record_last_idx + 1}")
if not os.path.isdir(best_model_dir_path):
    os.mkdir(best_model_dir_path)

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
    "init_layer": 9,
    "hidden_layer_1": 128,
    "hidden_layer_2": 64,
    "output_layer": 6
}

# instantiate the policy and its utilities
nn_policy = models.policy_methods_nn.SatColAvoidPolicy(conf=nn_conf).to(device=device)
policy_utils = PolicyMethodsUtils(observation_processing=data_preprocessing, device=device)
optimizer, optimizer_lr = policy_utils.instantiate_loss_fnc_optimiser(policy=nn_policy)

# set up training variables
train_eval_steps = 1000

train_total_losses = torch.tensor([], device=device)
train_rewards_sum_list = torch.tensor([], device=device)
eval_rewards_sum_list = torch.tensor([], device=device)
max_eval_reward_sum = -np.inf
best_model = copy.deepcopy(nn_policy)

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

    if eval_rewards_sum > max_eval_reward_sum:
        max_eval_reward_sum = eval_rewards_sum

        # save the best model
        best_model = copy.deepcopy(nn_policy)
        utils.save_best_model(best_model=best_model,
                              best_model_path=best_model_path,
                              best_model_dir_path=best_model_dir_path,
                              model_conf=nn_conf,
                              optimizer=optimizer,
                              optimizer_lr=optimizer_lr,
                              epoch=steps,
                              loss=train_losses[-1],
                              record_dict_path=policy_methods_record_dict_path,
                              model_record_dict=model_record_dict,
                              model_record_last_idx=model_record_last_idx,
                              max_eval_reward_sum=max_eval_reward_sum.item())

    print(f"{datetime.datetime.now()} - Epoch {steps} - Train Reward MeanSum: {train_rewards_sum} "
          f"| Eval Reward MeanSum: {eval_rewards_sum}")

    train_total_losses = torch.cat((train_total_losses, train_losses))
    train_rewards_sum_list = torch.cat((train_rewards_sum_list, torch.tensor([train_rewards_sum]).to(device=device)))
    eval_rewards_sum_list = torch.cat((eval_rewards_sum_list, torch.tensor([eval_rewards_sum]).to(device=device)))

n_loss = np.arange(0, len(train_total_losses))
n_rewards = np.arange(0, len(train_rewards_sum_list))

plt.plot(n_loss, train_total_losses.cpu())
plt.title(f"Losses")
plt.xlabel(f"Games played")
plt.ylabel(f"Loss value")
plt.grid()
plt.show()

plt.plot(n_rewards, train_rewards_sum_list.cpu())
plt.plot(n_rewards, eval_rewards_sum_list.cpu())
plt.title(f"Rewards received")
plt.xlabel(f"Games played")
plt.ylabel(f"Rewards received")
plt.grid()
plt.show()
