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
        plots_dir_path = self.create_plots_directory()

        for num_idx in range(num_runs):
            # play the game
            raw_rewards, final_info = self.play_game_once(game_env=game_env,
                                                          policy=policy)

            # store the statistics on the goals achieved by the model
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

            # create the evaluation plots
            self.create_evaluation_plots(game_final_info=final_info, plots_path_dir=plots_dir_path,
                                         plot_prefix=num_idx)

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

        return plots_dir_path

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

    def create_evaluation_plots(self, game_final_info, plots_path_dir, plot_prefix):
        # get the relevant raw data from the game info
        primary_init_seq = game_final_info["primary_init_sequence"]
        secondary_init_seq = game_final_info["secondary_init_sequence"]
        historical_actions = game_final_info["historical_actions"]
        historical_primary_sequence = game_final_info["historical_primary_sequence"]
        hist_primary_at_collision_states = game_final_info["hist_primary_at_collision_states"]
        collision_distance = game_final_info["collision_distance"]
        initial_orbit_radius_bound = game_final_info["initial_orbit_radius_bound"]
        max_altitude_diff_allowed = game_final_info["max_altitude_diff_allowed"]

        # get the difference between initial orbit states and final orbit states
        diff_init_final = self.compute_sequence_of_distances_between_state_seq(primary_init_seq,
                                                                               historical_primary_sequence)

        # get the difference between the primary and secondary satellites
        diff_primary_secondary = self.compute_sequence_of_distances_between_state_seq(hist_primary_at_collision_states,
                                                                                      secondary_init_seq)

        # get the policy action data
        action_means = [np.mean(x) for x in historical_actions]
        actions_x = [x[0] for x in historical_actions]
        actions_y = [x[1] for x in historical_actions]
        actions_z = [x[2] for x in historical_actions]

        # compute the plot for the differences between initial and final states
        init_final_plot, init_final_ax = plt.subplots()
        x_axis_data = np.arange(len(diff_init_final))

        init_final_ax.plot(x_axis_data, diff_init_final)
        init_final_ax.set_title("Differences between initial and modified orbit")
        init_final_plot.savefig(os.path.join(plots_path_dir, f"{plot_prefix}_Init_Final_Orbit_Diff.png"))

        # compute the plot for the differences between initial and final states
        collision_plot, collision_ax = plt.subplots()
        x_axis_data = np.arange(len(diff_primary_secondary))

        collision_ax.plot(x_axis_data, diff_primary_secondary)
        collision_ax.set_title("Differences between primary and secondary satellites")
        collision_plot.savefig(os.path.join(plots_path_dir, f"{plot_prefix}_Collision_Plot.png"))

        # compute the actions plots
        actions_plot, axs = plt.subplots(2, 3)
        x_axis_data = np.arange(len(action_means))

        axs[0, 1].plot(x_axis_data, action_means)
        axs[1, 0].plot(x_axis_data, actions_x)
        axs[1, 1].plot(x_axis_data, actions_y)
        axs[1, 2].plot(x_axis_data, actions_z)

        actions_plot.savefig(os.path.join(plots_path_dir, f"{plot_prefix}_Actions_Plot.png"))

    @staticmethod
    def compute_dist_between_states(primary_sc_state: np.array,
                                    secondary_sc_state: np.array) -> float:
        return np.linalg.norm(primary_sc_state - secondary_sc_state)

    @staticmethod
    def compute_sequence_of_distances_between_state_seq(primary_sc_state_seq: np.array,
                                                        secondary_sc_state_seq: np.array) -> np.array:
        seq_of_distances = []
        for idx, orb_pos_primary in enumerate(primary_sc_state_seq):
            seq_of_distances.append(np.linalg.norm(orb_pos_primary[:3] - secondary_sc_state_seq[idx][:3]))

        return np.array(seq_of_distances)
