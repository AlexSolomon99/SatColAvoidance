import copy
import os
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import datetime
import torch
import json
import shutil

import utils
import models
import dataprocessing
import model_evaluation.policy_evaluator as policy_evaluator
import model_evaluation.dqn_evaluator as dqn_evaluator
import ppo_alg.core as core

import sys

sys.path.append(r'E:\Alex\UniBuc\MasterThesis\gym-satellite-ca')

from gym_satellite_ca.envs import satDataClass


class PPOEvaluator(dqn_evaluator.DQNEvaluator):

    def __init__(self, device, sat_data_config, model_dir_path, model_file_path, model_evaluation_path):

        super().__init__(device, sat_data_config, model_dir_path, model_file_path, model_evaluation_path)

    def instantiate_model(self, device, model_dir_path, model_file_path, model_evaluation_path):
        if not os.path.isdir(model_dir_path):
            print(f"The model directory was not found - {model_dir_path}")
            sys.exit()

        if not os.path.isdir(model_evaluation_path):
            os.mkdir(model_evaluation_path)

        # loading the model config
        model_config_path = os.path.join(model_dir_path, "model_kwargs_conf.json")
        f = open(model_config_path, "r")
        model_conf = json.loads(f.read())
        f.close()

        # loading the model intended for evaluation
        checkpoint = torch.load(model_file_path)

        # # get observation dimension
        o, _ = self.game_env.reset()
        flat_state = self.data_preprocessing.transform_observations(game_env_obs=o)

        # Instantiate environment
        act_dim = self.game_env.action_space.shape

        # Create actor-critic module
        ac = core.MLPActorCritic(obs_dim=len(flat_state),
                                 action_space=self.game_env.action_space,
                                 **model_conf)

        ac.pi.load_state_dict(checkpoint['pi_model_state_dict'])
        ac.v.load_state_dict(checkpoint['vf_model_state_dict'])

        checkpoint_epoch = checkpoint['epoch']

        return ac, None, checkpoint_epoch, None

    def perform_evaluation(self, game_env, policy, num_runs=10, reset_options=None):
        # instantiate the dictionary containing the general status of the execution
        goals_overall_status_dict = {
            "num_runs": num_runs,
            "collision_avoided": 0,
            "returned_to_init_orbit": 0,
            "fuel_used_perc": 0.0,
            "raw_rewards_sum": 0.0,
            "individual_run_goals": {}
        }

        # create the plots directory
        plots_dir_path = self.create_plots_directory()

        for num_idx in range(num_runs):
            # play the game
            raw_rewards, final_info = self.play_game_once(game_env=game_env,
                                                          policy=policy)

            # store the statistics on the goals achieved by the model
            current_goals_status = {
                "collision_avoided": final_info["collision_avoided"],
                "returned_to_init_orbit": final_info["returned_to_init_orbit"],
                "fuel_used_perc": final_info["fuel_used_perc"],
                "raw_rewards": sum(raw_rewards.cpu().numpy().tolist())
            }
            goals_overall_status_dict["individual_run_goals"][num_idx] = current_goals_status
            goals_overall_status_dict = self.updated_goals_dict(goals_overall_status_dict,
                                                                current_goals_status)

            # create the evaluation plots
            self.create_evaluation_plots(game_final_info=final_info, plots_path_dir=plots_dir_path,
                                         plot_prefix=num_idx)

        utils.save_json(dict_=goals_overall_status_dict,
                        json_path=os.path.join(self.model_evaluation_path, "goals_evaluation.json"))

    def play_game_once(self, game_env, policy: core.MLPActorCritic):
        # setting the policy to the evaluation mode
        policy.eval()

        # resetting the game and getting the firs obs
        obs, _ = game_env.reset()
        final_info = None

        # condition for the game to be over
        done = False

        # set list of rewards
        raw_rewards = torch.tensor([], device=self.device)

        while not done:
            # transform the observations and perform inference
            flat_obs = self.data_preprocessing.transform_observations(game_env_obs=obs)
            model_obs = torch.from_numpy(flat_obs).to(device=self.device, dtype=torch.float32)
            action = policy.act(model_obs)

            obs, reward, done, truncated, info = game_env.step(action.tolist())
            done = done or truncated
            if done:
                final_info = info

            raw_rewards = torch.cat((raw_rewards, torch.tensor([reward]).to(device=self.device)))

        return raw_rewards, final_info
