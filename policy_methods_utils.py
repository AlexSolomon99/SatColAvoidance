import torch
import numpy as np

import dataprocessing
import models


class PolicyMethodsUtils:

    def __init__(self, observation_processing: dataprocessing.data_processing.ObservationProcessing,
                 device):
        self.observation_processing = observation_processing
        self.device = device

    def play_game_once(self, game_env, policy: torch.nn.Module, train: bool, optimizer):
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
        raw_rewards = torch.tensor([], device=self.device)
        log_probs = torch.tensor([], device=self.device)

        while not done:
            # transform the observations and perform inference
            flat_obs = self.observation_processing.transform_observations(game_env_obs=obs)
            model_obs = torch.from_numpy(flat_obs).to(device=self.device)
            action_parameters = policy(model_obs)

            # get the mean and std from the action parameters
            action_mean = action_parameters[:3]
            action_std = action_parameters[3:]

            # compute the action and the log prob from the normal distribution
            action_normal_distribution = torch.distributions.Normal(action_mean, action_std)
            action = action_normal_distribution.sample().to(device=self.device)
            log_prob = action_normal_distribution.log_prob(action).to(device=self.device)

            obs, reward, done, truncated, info = game_env.step(action.item())

            raw_rewards = torch.cat((raw_rewards, reward))
            log_probs = torch.cat((log_probs, log_prob.reshape(1)))

        # get the list of scores for each action
        scores_per_action = self.compute_discount_rate_reward(list_of_rewards=raw_rewards)

        if train:
            loss = self.update_policy(scores_per_action, log_probs, optimizer)
        else:
            loss = 0

        return loss, scores_per_action, raw_rewards

    @staticmethod
    def instantiate_loss_fnc_optimiser(policy: torch.nn.Module, lr=0.01):
        optimizer = torch.optim.Adam(lr=lr, params=policy.parameters())

        return optimizer

    def train_policy(self, policy: torch.nn.Module, game_env, optimizer, num_train_iterations=500):
        policy.train()

        list_of_losses = torch.tensor([], device=self.device)
        list_of_raw_rewards = torch.tensor([], device=self.device)

        for train_counter in range(num_train_iterations):
            loss, scores_per_action, raw_rewards = self.play_game_once(game_env=game_env,
                                                                       policy=policy,
                                                                       train=True,
                                                                       optimizer=optimizer)

            list_of_losses = torch.cat((list_of_losses, loss))
            list_of_raw_rewards = torch.cat((list_of_raw_rewards, raw_rewards))

        return list_of_losses, list_of_raw_rewards

    def eval_policy(self, policy: torch.nn.Module, game_env, optimizer, num_eval_iterations=500):
        policy.eval()

        list_of_raw_rewards = torch.tensor([], device=self.device)

        for train_counter in range(num_eval_iterations):
            loss, scores_per_action, raw_rewards = self.play_game_once(game_env=game_env,
                                                                       policy=policy,
                                                                       train=False,
                                                                       optimizer=optimizer)

            list_of_raw_rewards = torch.cat((list_of_raw_rewards, raw_rewards))

        return list_of_raw_rewards

    def compute_discount_rate_reward(self, list_of_rewards: torch.tensor, discount_rate=0.95):
        scores_list = torch.tensor([], device=self.device)
        for idx, reward in enumerate(list_of_rewards):
            current_score = self.compute_individual_score(list_of_rewards=list_of_rewards[idx:].copy(),
                                                          discount_rate=discount_rate)
            scores_list = torch.cat((scores_list, current_score))

        return torch.tensor(scores_list)

    def compute_individual_score(self, list_of_rewards: torch.tensor, discount_rate: float) -> torch.Tensor:
        """
        Function which computes the score of the current action, given as input the list of rewards starting from the
        current action onwards and the discount rate

        :param list_of_rewards: List of rewards (the reward of the current action is the first one)
        :param discount_rate: The rate at which each reward of the consecutive action is being taken into consideration
        :return: Reward score of the action

        score_0 = reward_0 * r^0 + reward_1 * r^1 + reward_2 * r^2
        """
        score = 0
        for idx, elem in enumerate(list_of_rewards):
            score += elem.item() * discount_rate ** idx
        return torch.Tensor([score], device=self.device)

    @staticmethod
    def update_policy(returns, log_prob_actions, optimizer):
        returns = returns.detach()

        loss = - (returns * log_prob_actions).sum()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        return loss.item()
