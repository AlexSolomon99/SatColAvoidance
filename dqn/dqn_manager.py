import os
import numpy as np
import torch
import gymnasium as gym
import copy
import datetime

from models.dqn_nn import QNetwork
from dqn_replay_memory import ReplayMemory
import utils
import dataprocessing
import dqn_utils

import sys

sys.path.append(r'E:\Alex\UniBuc\MasterThesis\gym-satellite-ca')

from gym_satellite_ca.envs import satDataClass

# set device
device = torch.device('cuda')

# model name
MODEL_NAME = "dqn_model"

# constants
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# constant paths
BASE = r"E:\Alex\UniBuc\MasterThesis\src"
DATA_PATH = os.path.join(BASE, "data")
RECORDED_MODELS_PATH = os.path.join(BASE, "recorded_models")
DQN_METHODS_MODELS = os.path.join(RECORDED_MODELS_PATH, "dqn_models")

if not os.path.isdir(RECORDED_MODELS_PATH):
    os.mkdir(RECORDED_MODELS_PATH)

if not os.path.isdir(DQN_METHODS_MODELS):
    os.mkdir(DQN_METHODS_MODELS)

# paths - set up the model record dictionary
sat_data_config = os.path.join(DATA_PATH, "default_sat_data_config.json")
dqn_record_dict_path = os.path.join(DQN_METHODS_MODELS, "dqn_record.json")

if not os.path.isfile(dqn_record_dict_path):
    model_record_dict = {}
    model_record_last_idx = 0
else:
    model_record_dict = utils.read_json(json_path=dqn_record_dict_path)
    model_record_last_idx = len(model_record_dict.keys())
best_model_path = os.path.join(DQN_METHODS_MODELS, f"{MODEL_NAME}_{model_record_last_idx + 1}")

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
    "init_layer": 6968,
    "hidden_layer_1": 1000,
    "hidden_layer_2": 100,
    "output_layer": 3
}

# create a policy network and a target network, which is a mirror of it
policy_net = QNetwork(nn_conf).to(device)
target_net = QNetwork(nn_conf).to(device)
target_net.load_state_dict(policy_net.state_dict())

# set up the optimizer
optimizer = torch.optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

# setting up the replay memory
memory = ReplayMemory(capacity=10000)

# set up the utilities class
dqn_utils_class = dqn_utils.DQNUtils(observation_processing=data_preprocessing, memory=memory, optimizer=optimizer,
                                     device=device, batch_size=BATCH_SIZE, gamma=GAMMA,
                                     eps_start=EPS_START, eps_end=EPS_END, eps_decay=EPS_DECAY, tau=TAU)

# set up training variables
steps_done = 0
num_episodes = 100
max_eval_reward_sum = -np.inf
best_policy = copy.deepcopy(policy_net)

print(f"{datetime.datetime.now()} - Started training")
for i_episode in range(num_episodes):
    print(f"{datetime.datetime.now()} - Episode {i_episode}")
    steps_done, raw_rewards = dqn_utils_class.play_game_once(game_env=env,
                                                             policy_net=policy_net,
                                                             target_net=target_net,
                                                             steps_done=steps_done)
    rewards_sum = raw_rewards.sum()
    print(f"{datetime.datetime.now()} - Epoch {i_episode} - Train Reward MeanSum: {rewards_sum.item()}")

    if rewards_sum.item() > max_eval_reward_sum:
        max_eval_reward_sum = rewards_sum.item()

        # save the best model
        best_model = copy.deepcopy(policy_net)
        utils.save_best_model(best_model=best_model,
                              best_model_path=best_model_path,
                              record_dict_path=dqn_record_dict_path,
                              model_record_dict=model_record_dict,
                              model_record_last_idx=model_record_last_idx,
                              max_eval_reward_sum=max_eval_reward_sum.item())

