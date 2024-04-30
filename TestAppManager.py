import gymnasium
import sys
import datetime
import torch

import numpy as np
import os

import utils
from dataprocessing import data_processing

sys.path.append(r'E:\Alex\UniBuc\MasterThesis\gym-satellite-ca')

# constant paths
BASE = "./"
DATA_PATH = os.path.join(BASE, "data")

# paths
sat_data_config = os.path.join(DATA_PATH, "default_sat_data_config.json")

# setting up the satellite data and init config of the environment
init_sat = utils.get_sat_data_env(sat_data_config)

env = gymnasium.make('gym_satellite_ca:gym_satellite_ca/CollisionAvoidance-v0', satellite=init_sat)

print("Random action: ", env.action_space.sample())

x = torch.tensor(env.action_space.sample(), dtype=torch.float)
print(x)

sys.exit()

print(env.observation_space)
print(env.action_space)

obs, _ = env.reset()
print("TCA Time Lapse shape", obs['tca_time_lapse'].shape)
print("Primary mass", obs['primary_sc_mass'])
print("Primary Sc Seq shape", obs['primary_sc_state_seq'].shape)
print("Secondary Seq shape", obs['secondary_sc_state_seq'].shape)

print(f"Primary current pos: {obs['primary_current_pv']}")

tca_time_lapse_max_abs_val = env.observation_space['tca_time_lapse'].high[0]
data_preprocessing = data_processing.ObservationProcessing(satellite_data=env.unwrapped.satellite,
                                                           tca_time_lapse_max_abs_val=tca_time_lapse_max_abs_val)

flattened_data = data_preprocessing.transform_observations(game_env_obs=obs)
print(f"Flattened obs: {flattened_data}")

sys.exit()

# print(np.min(obs['primary_sc_state_seq']))
# print(np.max(obs['primary_sc_state_seq']))
# print(np.min(obs['secondary_sc_state_seq']))
# print(np.max(obs['secondary_sc_state_seq']))

print(f"Start game: {datetime.datetime.now()}")
for _ in range(2):
    rewards, observations, actions = utils.play_constant_game_manually(game_env=env, constant_action=[0.0, 0.0, 0.0])
    print(f"Game finished")
    print(f"List of rewards: {rewards}")

    print(f"Sum of rewards: {np.sum(rewards)}")
    print()

