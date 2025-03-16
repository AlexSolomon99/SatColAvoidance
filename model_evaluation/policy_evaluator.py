import os
import torch
import json

import utils
import models
import model_evaluation.dqn_evaluator as dqn_evaluator


class PolicyEvaluator(dqn_evaluator.DQNEvaluator):

    def __init__(self, device, sat_data_config, model_dir_path, model_file_path, model_evaluation_path):

        super().__init__(device, sat_data_config, model_dir_path, model_file_path, model_evaluation_path)

    @staticmethod
    def instantiate_model(device, model_dir_path, model_file_path, model_evaluation_path):
        if not os.path.isdir(model_dir_path):
            print(f"The model directory was not found - {model_dir_path}")
            exit()

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
            # flat_obs = self.data_preprocessing.transform_observations(game_env_obs=obs)
            model_obs = torch.from_numpy(obs).to(device=self.device, dtype=torch.float)
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

