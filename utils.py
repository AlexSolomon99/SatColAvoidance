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


def play_game_once(game_env, policy: torch.nn.Module, train: bool, optimizer):
    # setting the policy to the training mode
    if train:
        policy.train()
    else:
        policy.eval()

    # resetting the game and getting the firs obs
    obs, _ = game_env.reset()

    # condition for the game to be over
    done = False

    # set list of rewards
    rewards = []
    log_probs = []

    while not done:
        possible_actions = policy(torch.Tensor(obs))
        action_probabilities = softmax(possible_actions)

        categorical_action = torch.distributions.Categorical(action_probabilities)
        action = categorical_action.sample()
        log_prob = categorical_action.log_prob(action)

        obs, reward, done, truncated, info = game_env.step(action.item())

        rewards.append(reward)
        log_probs.append(log_prob.reshape(1))

    # print(f"LOF PROBABILITIES: {log_probabilities}")
    log_probabilities = torch.cat(log_probs)

    # get the list of scores for each action
    scores_per_action = NU_NeuralUtils.NeuralNetUtils().compute_discount_rate_reward(list_of_rewards=rewards)

    if train:
        loss = NU_NeuralUtils.NeuralNetUtils().update_policy(scores_per_action, log_probabilities, optimizer)
    else:
        loss = 0

    return loss, len(rewards), scores_per_action
