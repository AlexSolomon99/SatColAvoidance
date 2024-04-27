import datetime
import json
import copy
import torch

import sys

sys.path.append(r'E:\Alex\UniBuc\MasterThesis\gym-satellite-ca')

from gym_satellite_ca.envs import satDataClass


def read_config_json(config_path: str):
    f = open(config_path, "r")
    data_config = json.load(f)
    f.close()

    return data_config


def get_sat_data_env(sat_data_config_path: str):
    sat_data_config = read_config_json(config_path=sat_data_config_path)

    iss_satellite = satDataClass.SatelliteData(**sat_data_config)
    iss_satellite.change_angles_to_radians()
    iss_satellite.set_random_tran()

    return copy.deepcopy(iss_satellite)


def play_game_manually(game_env):

    # resetting the game and getting the firs obs
    obs, _ = game_env.reset()

    # condition for the game to be over
    done = False

    # set list of rewards and observations
    rewards = []
    observations = []
    actions = []

    while not done:
        print(f"Step idx: {game_env.unwrapped.time_step_idx}")
        action = list(map(int, input("\nCurrent Action : ").strip().split()))[:game_env.action_space.shape[0]]
        print("Action received from user: ", action)

        obs, reward, terminated, truncated, info = game_env.step(action)

        rewards.append(reward)
        actions.append(action)
        observations.append(obs)

    return rewards, observations, actions


def play_constant_game_manually(game_env, constant_action):

    # resetting the game and getting the firs obs
    obs, _ = game_env.reset()

    # condition for the game to be over
    done = False

    # set list of rewards and observations
    rewards = []
    observations = []
    actions = []

    curr_idx = 0

    while not done:
        curr_idx += 1
        if curr_idx % 100 == 0:
            print(f"{datetime.datetime.now()} - Current step idx: ", game_env.unwrapped.time_step_idx)

        action = constant_action
        obs, reward, done, truncated, info = game_env.step(action)

        rewards.append(reward)
        actions.append(action)
        observations.append(obs)

    return rewards, observations, actions

