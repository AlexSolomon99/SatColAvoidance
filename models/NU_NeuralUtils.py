import torch


class NeuralNetUtils:

    def __init__(self, game_env, game_utils):
        self.game_env = game_env
        self.game_utils = game_utils

    @staticmethod
    def instantiate_loss_fnc_optimiser(policy: torch.nn.Module, lr=0.01):
        # loss function and optimizer definition
        # TODO: why Adam
        optimizer = torch.optim.Adam(lr=lr, params=policy.parameters())

        return optimizer

    def train_policy(self, policy: torch.nn.Module, game_env, optimizer, num_train_iterations=500):
        policy.train()

        iterations_performed = []
        list_of_losses = []

        for train_counter in range(num_train_iterations):
            loss, num_iterations, scores_per_action = self.game_utils.play_game_once(game_env=game_env,
                                                                                     policy=policy,
                                                                                     train=True,
                                                                                     optimizer=optimizer)

            iterations_performed.append(num_iterations)
            list_of_losses.append(loss)

        return list_of_losses, iterations_performed

    def eval_policy(self, policy: torch.nn.Module, game_env, optimizer, num_eval_iterations=500):
        policy.eval()

        iterations_performed = []
        list_of_losses = []

        for train_counter in range(num_eval_iterations):
            loss, num_iterations, scores_per_action = self.game_utils.play_game_once(game_env=game_env,
                                                                                     policy=policy,
                                                                                     train=False,
                                                                                     optimizer=optimizer)

            iterations_performed.append(num_iterations)

        return iterations_performed

    def compute_discount_rate_reward(self, list_of_rewards: list, discount_rate=0.95):
        scores_list = []
        for idx, reward in enumerate(list_of_rewards):
            current_score = self.compute_individual_score(list_of_rewards=list_of_rewards[idx:].copy(),
                                                          discount_rate=discount_rate)
            scores_list.append(current_score)

        return torch.tensor(scores_list)

    def compute_individual_score(self, list_of_rewards: list, discount_rate: float) -> float:
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
            score += elem * discount_rate ** idx
        return score

    @staticmethod
    def update_policy(returns, log_prob_actions, optimizer):
        returns = returns.detach()

        loss = - (returns * log_prob_actions).sum()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        return loss.item()
