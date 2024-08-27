import os
import datetime
import json
import copy
import torch
import gymnasium as gym
import dataprocessing

import sys

sys.path.append(r'E:\Alex\UniBuc\MasterThesis\gym-satellite-ca')

from gym_satellite_ca.envs import satDataClass


def read_json(json_path: str):
    f = open(json_path, "r")
    data_config = json.load(f)
    f.close()

    return data_config


def save_json(dict_: dict, json_path: str):
    with open(json_path, "w") as outfile:
        json.dump(dict_, outfile, indent=4)


def get_sat_data_env(sat_data_config_path: str):
    sat_data_config = read_json(json_path=sat_data_config_path)

    iss_satellite = satDataClass.SatelliteData(**sat_data_config)
    iss_satellite.change_angles_to_radians()
    iss_satellite.set_random_tran()

    return copy.deepcopy(iss_satellite)


def set_up_environment(sat_data_config):
    # setting up the satellite data and init config of the environment
    init_sat = get_sat_data_env(sat_data_config)

    # setting up the environment
    env = gym.make('gym_satellite_ca:gym_satellite_ca/CollisionAvoidance-v0',
                   satellite=init_sat)

    # set up the observation processing class
    tca_time_lapse_max_abs_val = env.observation_space['tca_time_lapse'].high[0]
    data_preprocessing = dataprocessing.data_processing.ObservationProcessing(
        satellite_data=env.unwrapped.satellite,
        tca_time_lapse_max_abs_val=tca_time_lapse_max_abs_val)

    return env, data_preprocessing


def set_up_kepl_environment(sat_data_config):
    # setting up the satellite data and init config of the environment
    init_sat = get_sat_data_env(sat_data_config)

    # setting up the environment
    env = gym.make('gym_satellite_ca:gym_satellite_ca/CollisionAvoidance-v0',
                   satellite=init_sat)

    return env


def save_best_model(best_model: torch.nn.Module,
                    best_model_path: str,
                    best_model_dir_path: str,
                    model_conf: dict,
                    optimizer,
                    optimizer_lr: float,
                    epoch: int,
                    loss,
                    record_dict_path: str,
                    model_record_dict: dict,
                    model_record_last_idx: int,
                    max_eval_reward_sum: int):
    torch.save({
        'epoch': epoch,
        'model_state_dict': best_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'optimizer_lr': optimizer_lr
    }, best_model_path)
    save_json(dict_=model_conf, json_path=os.path.join(best_model_dir_path, "model_conf.json"))

    model_record_dict[model_record_last_idx + 1] = {
        'path': best_model_path,
        'max_eval_reward_sum': max_eval_reward_sum
    }
    save_json(dict_=model_record_dict, json_path=record_dict_path)


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


def play_constant_game_manually(game_env, constant_action, reset_options=None):

    # resetting the game and getting the firs obs
    obs, _ = game_env.reset(options=reset_options)

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
        # if curr_idx == 4:
        #     sys.exit()
        # print(f"Current observation: {obs}")
        # print()

        done = done or truncated

        rewards.append(reward)
        actions.append(action)
        observations.append(obs)

    return rewards, observations, actions, info

