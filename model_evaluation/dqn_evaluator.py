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


class DQNEvaluator:

    ACTION_SPACE = [-1, 0, 1]

    def __init__(self, device, sat_data_config, model_dir_path, model_file_path,
                 model_evaluation_path):

        self.device = device

        # environment setup attributes
        self.sat_data_config = sat_data_config
        self.game_env, self.data_preprocessing = self.set_up_environment(sat_data_config=self.sat_data_config)
        self.full_action_space = self.get_full_action_space()

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

        policy = models.dqn_nn.QNetwork(conf=model_conf).to(device=device)
        optimizer = torch.optim.AdamW(lr=checkpoint['optimizer_lr'], params=policy.parameters())

        policy.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        checkpoint_epoch = checkpoint['epoch']
        checkpoint_loss = checkpoint['loss']

        return policy, optimizer, checkpoint_epoch, checkpoint_loss

    def play_game_once(self, game_env, policy: torch.nn.Module, reset_options=None):
        # setting the policy to the evaluation mode
        policy.eval()

        # resetting the game and getting the firs obs
        obs, _ = game_env.reset(options=reset_options)
        final_info = None

        # condition for the game to be over
        done = False

        # set list of rewards
        raw_rewards = torch.tensor([], device=self.device)

        while not done:
            # transform the observations and perform inference
            flat_obs = self.data_preprocessing.transform_observations(game_env_obs=obs)
            model_obs = torch.from_numpy(flat_obs).to(device=self.device, dtype=torch.float)

            # select an action
            action, action_idx = (self.full_action_space[policy(model_obs).argmax()],
                                  policy(model_obs).argmax().view(1, 1))

            # apply the action and get the next state
            obs, reward, done, truncated, info = game_env.step(action.tolist())
            done = done or truncated
            if done:
                final_info = info

            raw_rewards = torch.cat((raw_rewards, torch.tensor([reward]).to(device=self.device)))

        return raw_rewards, final_info

    def get_full_action_space(self):
        all_actions = []

        for elem_1 in self.ACTION_SPACE:
            action_1 = [elem_1]
            for elem_2 in self.ACTION_SPACE:
                action_2 = action_1 + [elem_2]
                for elem_3 in self.ACTION_SPACE:
                    current_action = action_2 + [elem_3]
                    all_actions.append(torch.tensor(current_action, device=self.device, dtype=torch.float))

        return all_actions

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
            print(f"Step {num_idx}")
            # play the game
            raw_rewards, final_info = self.play_game_once(game_env=game_env,
                                                          policy=policy,
                                                          reset_options=reset_options)

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
        goals_overall_status_dict["fuel_used_perc"] += (current_goals_status["fuel_used_perc"] /
                                                        goals_overall_status_dict["num_runs"])
        goals_overall_status_dict["raw_rewards_sum"] += (current_goals_status["raw_rewards"] /
                                                         goals_overall_status_dict["num_runs"])

        return goals_overall_status_dict

    def create_evaluation_plots(self, game_final_info, plots_path_dir, plot_prefix):
        # get the relevant raw data from the game info
        init_kepl_elements = game_final_info["init_kepl_elements"]

        historical_actions = game_final_info["historical_actions"]
        hist_kepl_elements = game_final_info["hist_kepl_elements"]

        collision_idx = game_final_info["collision_idx"]

        # get the difference between the historical keplerian elements and the initial values
        diff_sma, diff_ecc, diff_inc, diff_par, diff_ran = self.get_diff_between_hist_kepl_elem_and_initial_ones(
            historical_kepl_elem=hist_kepl_elements, init_kepl_elem=init_kepl_elements
        )
        # normalise some keplerian elements
        diff_ecc = [1e6 * x for x in diff_ecc]
        diff_inc = [1e5 * x for x in diff_inc]
        diff_par = [1e5 * x for x in diff_par]
        diff_ran = [1e5 * x for x in diff_ran]

        # get the policy action data
        action_means = [np.mean(x) for x in historical_actions]
        actions_x = [x[0] for x in historical_actions]
        actions_y = [x[1] for x in historical_actions]
        actions_z = [x[2] for x in historical_actions]

        # compute the actions plots
        actions_plot, axs = plt.subplots(1, 3, figsize=(15, 5))
        x_axis_data = np.arange(len(action_means))
        x_label = 'Time steps [x10 min]'

        axs[0].plot(x_axis_data, actions_x, color='r')
        axs[1].plot(x_axis_data, actions_y, color='g')
        axs[2].plot(x_axis_data, actions_z, color='b')

        # Add labels and titles
        axs[0].set_xlabel(x_label)
        axs[0].set_ylabel('Thrust Level [x1 mN]')
        axs[0].set_title('Ox Direction')

        axs[1].set_xlabel(x_label)
        axs[1].set_title('Oy Direction')

        axs[2].set_xlabel(x_label)
        axs[2].set_title('Oz Direction')

        # actions_plot.suptitle('Thrust Levels over time (VNC Frame)', fontsize=16)
        for ax_idx in range(3):
            axs[ax_idx].grid(True)
            axs[ax_idx].axvline(x=collision_idx, color='k', linestyle='--', label="TCA")
            axs[ax_idx].legend(loc='upper right')

        actions_plot.savefig(os.path.join(plots_path_dir, f"{plot_prefix}_Actions_Plot.png"))
        plt.close(actions_plot)

        # compute the keplerian elements plots
        keplerian_plot, axs = plt.subplots(5, 1, figsize=(25, 20))
        x_axis_data = np.arange(len(diff_sma))
        x_label = 'Time steps [x10 min]'

        # plot the data
        axs[0].plot(x_axis_data, diff_sma, color='navy')
        axs[1].plot(x_axis_data, diff_ecc, color='darkviolet')
        axs[2].plot(x_axis_data, diff_inc, color='darkred')
        axs[3].plot(x_axis_data, diff_par, color='b')
        axs[4].plot(x_axis_data, diff_ran, color='g')

        # Add labels and titles
        label_pad = 24
        ylabel_fontsize = 22
        axs[0].set_ylabel('Semi-major-axis [m]', fontsize=ylabel_fontsize, labelpad=label_pad)
        axs[1].set_ylabel('Eccentricity \n [x1e-6]', fontsize=ylabel_fontsize, labelpad=label_pad)
        axs[2].set_ylabel('Inclination \n [x1e-5 rad]', fontsize=ylabel_fontsize, labelpad=label_pad)
        axs[3].set_ylabel('Perigee \n Argument [1e-5 rad]', fontsize=ylabel_fontsize, labelpad=label_pad)
        axs[4].set_ylabel('RAAN [x1e-5 rad]', fontsize=ylabel_fontsize, labelpad=label_pad)

        # keplerian_plot.suptitle('Keplerian Elements Differences over Time', fontsize=36)
        for ax_idx in range(5):
            if ax_idx == 4:
                axs[ax_idx].set_xlabel(x_label, fontsize=22)
            axs[ax_idx].grid(True)
            axs[ax_idx].axvline(x=collision_idx, color='k', linestyle='--', label="TCA")
            axs[ax_idx].tick_params(axis='both', labelsize=18)
            if ax_idx == 0:
                axs[ax_idx].legend(loc='upper right', fontsize=32)

        keplerian_plot.savefig(os.path.join(plots_path_dir, f"{plot_prefix}_Keplerian_Diff.png"))
        plt.close(keplerian_plot)

        # save a json with the plotted data
        dict_plotted_data = {
            "historical_actions": historical_actions,
            "sma_hist": diff_sma,
            "ecc_hist": diff_ecc,
            "inc_hist": diff_inc,
            "pa_hist": diff_par,
            "raan_hist": diff_ran,
            "init_kepl_elements": init_kepl_elements,
            "collision_idx": collision_idx
        }
        utils.save_json(dict_=dict_plotted_data,
                        json_path=os.path.join(plots_path_dir, f"{plot_prefix}_plotted_data.json"))

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

    @staticmethod
    def get_diff_between_hist_kepl_elem_and_initial_ones(historical_kepl_elem, init_kepl_elem):
        num_comp_steps = len(historical_kepl_elem)
        diff_sma = [historical_kepl_elem[x][0] - init_kepl_elem[0] for x in range(num_comp_steps)]
        diff_ecc = [historical_kepl_elem[x][1] - init_kepl_elem[1] for x in range(num_comp_steps)]
        diff_inc = [historical_kepl_elem[x][2] - init_kepl_elem[2] for x in range(num_comp_steps)]
        diff_par = [min(abs(historical_kepl_elem[x][3] - init_kepl_elem[3]),
                        abs(2*np.pi - abs((historical_kepl_elem[x][3] - init_kepl_elem[3])))) for x in range(num_comp_steps)]
        diff_ran = [historical_kepl_elem[x][4] - init_kepl_elem[4] for x in range(num_comp_steps)]

        return diff_sma, diff_ecc, diff_inc, diff_par, diff_ran

