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
RESET_OPTIONS = {
    "propagator": "numerical",
    "generate_sat": False
}

# paths
sat_data_config = os.path.join(DATA_PATH, "default_sat_data_config.json")

# setting up the satellite data and init config of the environment
init_sat = utils.get_sat_data_env(sat_data_config)

env = gymnasium.make('gym_satellite_ca:gym_satellite_ca/CollisionAvoidance-v0', satellite=init_sat)
obs, _ = env.reset(options=RESET_OPTIONS)

# tca_time_lapse_max_abs_val = env.observation_space['tca_time_lapse'].high[0]
# data_preprocessing = data_processing.ObservationProcessing(satellite_data=env.unwrapped.satellite,
#                                                            tca_time_lapse_max_abs_val=tca_time_lapse_max_abs_val)
#
# flattened_data = data_preprocessing.transform_observations(game_env_obs=obs)
# print(f"Flattened obs: {flattened_data}")

print(f"Start game: {datetime.datetime.now()}")
for _ in range(1):
    rewards, observations, actions, info = utils.play_constant_game_manually(game_env=env, constant_action=[0.0, 0.0, 0.0],
                                                                             reset_options=RESET_OPTIONS)
    # print(f"Game finished")
    # print(f"List of rewards: {rewards}")
    #
    # print(f"Sum of rewards: {np.sum(rewards)}")
    # print(f"Info: {info}")
    # print()
print(f"Game ended: {datetime.datetime.now()}")

