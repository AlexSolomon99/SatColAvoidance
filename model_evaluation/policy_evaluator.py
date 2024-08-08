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

import sys

sys.path.append(r'E:\Alex\UniBuc\MasterThesis\gym-satellite-ca')

from gym_satellite_ca.envs import satDataClass


class PolicyEvaluator:

    def __init__(self, device, sat_data_config, model_dir_path, model_file_path,
                 model_evaluation_path):

        self.device = device

        # environment setup attributes
        self.sat_data_config = sat_data_config
        self.game_env, self.data_preprocessing = self.set_up_environment(sat_data_config=self.sat_data_config)

        # model setup attributes
        self.model_dir_path = model_dir_path
        self.model_evaluation_path = model_evaluation_path
        self.model_file_path = model_file_path
        self.policy, self.optimizer, self.checkpoint_epoch, self.checkpoint_loss = (
            self.instantiate_model(self.device, self.model_dir_path, self.model_file_path, self.model_evaluation_path))

    @staticmethod
    def set_up_environment(sat_data_config):
        # setting up the satellite data and init config of the environment
        init_sat = utils.get_sat_data_env(sat_data_config)

        # setting up the environment
        env = gym.make('gym_satellite_ca:gym_satellite_ca/CollisionAvoidance-v0',
                       satellite=init_sat)

        # set up the observation processing class
        tca_time_lapse_max_abs_val = env.observation_space['tca_time_lapse'].high[0]
        data_preprocessing = dataprocessing.data_processing.ObservationProcessing(
            satellite_data=env.unwrapped.satellite,
            tca_time_lapse_max_abs_val=tca_time_lapse_max_abs_val)

        return env, data_preprocessing

    @staticmethod
    def instantiate_model(device, model_dir_path, model_file_path, model_evaluation_path):
        if not os.path.isdir(model_dir_path):
            print(f"The model directory was not found - {model_dir_path}")
            sys.exit()

        if not os.path.isdir(model_evaluation_path):
            os.mkdir(model_evaluation_path)

        # loading the model config
        model_config_path = os.path.join(model_dir_path, "model_conf.json")
        f = open(model_config_path, "r")
        model_conf = json.loads(f.read())
        f.close()

        # loading the model intended for evaluation
        checkpoint = torch.load(model_file_path)

        policy = models.policy_methods_nn.SatColAvoidPolicy(conf=model_conf).to(device=device)
        optimizer = torch.optim.Adam(lr=checkpoint['optimizer_lr'], params=policy.parameters())

        policy.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        checkpoint_epoch = checkpoint['epoch']
        checkpoint_loss = checkpoint['loss']

        return policy, optimizer, checkpoint_epoch, checkpoint_loss

    def play_game_once(self, game_env, policy: torch.nn.Module):
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
            model_obs = torch.from_numpy(flat_obs).to(device=self.device, dtype=torch.float)
            action_parameters = policy(model_obs)

            # get the mean and std from the action parameters
            action_mean = action_parameters[:3]
            action_std = action_parameters[3:]

            # compute the action and the log prob from the normal distribution
            action_normal_distribution = torch.distributions.Normal(action_mean, action_std)
            action = action_normal_distribution.sample().to(device=self.device)

            obs, reward, done, truncated, info = game_env.step(action.tolist())
            if done:
                final_info = info

            raw_rewards = torch.cat((raw_rewards, torch.tensor([reward]).to(device=self.device)))

        return raw_rewards, final_info

    def perform_evaluation(self, game_env, policy, num_runs=10):
        # instantiate the dictionary containing the general status of the execution
        goals_overall_status_dict = {
            "num_runs": num_runs,
            "collision_avoided": 0,
            "returned_to_init_orbit": 0,
            "drifted_out_of_bounds": 0,
            "fuel_used_perc": 0.0,
            "raw_rewards_sum": 0.0,
            "individual_run_goals": {}
        }

        # create the plots directory
        self.create_plots_directory()

        for num_idx in range(num_runs):
            raw_rewards, final_info = self.play_game_once(game_env=game_env,
                                                          policy=policy)
            current_goals_status = {
                "collision_avoided": final_info["collision_avoided"],
                "returned_to_init_orbit": final_info["returned_to_init_orbit"],
                "drifted_out_of_bounds": final_info["drifted_out_of_bounds"],
                "fuel_used_perc": final_info["fuel_used_perc"],
                "raw_rewards": sum(raw_rewards.cpu().numpy().tolist())
            }
            goals_overall_status_dict["individual_run_goals"].update(current_goals_status)
            goals_overall_status_dict = self.updated_goals_dict(goals_overall_status_dict,
                                                                current_goals_status)

        utils.save_json(dict_=goals_overall_status_dict,
                        json_path=os.path.join(self.model_evaluation_path, "goals_evaluation.json"))

    def create_plots_directory(self):
        plots_dir_path = os.path.join(self.model_evaluation_path, "evaluation_plots")

        # if the directory does not exist, create it, otherwise delete its content
        if not os.path.isdir(plots_dir_path):
            os.mkdir(plots_dir_path)
        else:
            for filename in os.listdir(plots_dir_path):
                file_path = os.path.join(plots_dir_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))

    @staticmethod
    def updated_goals_dict(goals_overall_status_dict, current_goals_status):
        if current_goals_status["collision_avoided"] is True:
            goals_overall_status_dict["collision_avoided"] += 1
        if current_goals_status["returned_to_init_orbit"] is True:
            goals_overall_status_dict["returned_to_init_orbit"] += 1
        if current_goals_status["drifted_out_of_bounds"] is True:
            goals_overall_status_dict["drifted_out_of_bounds"] += 1
        goals_overall_status_dict["fuel_used_perc"] += (current_goals_status["fuel_used_perc"] /
                                                        goals_overall_status_dict["num_runs"])
        goals_overall_status_dict["raw_rewards_sum"] += (current_goals_status["raw_rewards"] /
                                                         goals_overall_status_dict["num_runs"])

        return goals_overall_status_dict
