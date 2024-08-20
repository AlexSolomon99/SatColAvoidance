import torch
import random
import math
import copy
from itertools import count

from dqn_replay_memory import Transition


class DQNUtils:

    def __init__(self, memory, optimizer, device, eps_start: float, eps_end: float,
                 eps_decay: float,
                 tau: float, gamma: float, batch_size: int, local_action_space: list, observation_processing=None):
        self.observation_processing = observation_processing
        self.memory = memory
        self.optimizer = optimizer
        self.device = device
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        self.batch_size = batch_size
        self.gamma = gamma
        self.local_action_space = local_action_space
        self.full_action_space = self.get_full_action_space()

    def play_game_once(self, game_env, policy_net: torch.nn.Module, target_net: torch.nn.Module,
                       steps_done: int, reset_options: dict):

        # initialise the reward tensor
        raw_rewards = torch.tensor([], device=self.device, dtype=torch.float)
        losses_tensor = torch.tensor([], device=self.device, dtype=torch.float)

        # initialize the environment and get its state
        obs, _ = game_env.reset(options=reset_options)
        # transform the observations
        # flat_state = self.observation_processing.transform_observations(game_env_obs=obs)
        state = torch.from_numpy(obs).to(device=self.device, dtype=torch.float)

        for t in count():
            steps_done += 1

            # select an action
            action, action_idx = self.select_action(steps_done=steps_done, policy_net=policy_net,
                                                    _state=state,
                                                    eps_start=self.eps_start, eps_end=self.eps_end,
                                                    eps_decay=self.eps_decay)

            # apply the action and get the reward and next state
            obs, reward, done, truncated, info = game_env.step(action.tolist())
            reward = torch.tensor([reward], device=self.device, dtype=torch.float)
            done = done or truncated

            raw_rewards = torch.cat((raw_rewards, reward))

            if done:
                next_state = None
                next_state_unsqueezed = None
            else:
                # next_flat_state = self.observation_processing.transform_observations(game_env_obs=obs)
                next_state = torch.from_numpy(obs).to(device=self.device, dtype=torch.float)
                next_state_unsqueezed = next_state.unsqueeze(0)

            # push the transition to the memory
            self.memory.push(state.unsqueeze(0), action_idx, next_state_unsqueezed, reward)
            state = copy.deepcopy(next_state)

            # optimise the model
            loss = self.optimize_model(policy_net=policy_net, target_net=target_net)
            if loss is not None:
                loss_item = loss.item()
                losses_tensor = torch.cat((losses_tensor, torch.tensor([loss_item], device=self.device)))

            # soft update of the target network's weights once every 1000 steps
            if (steps_done + 1) % 1000 == 0:
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = (policy_net_state_dict[key] * self.tau + target_net_state_dict[key] *
                                                  (1 - self.tau))
                target_net.load_state_dict(target_net_state_dict)

            if done:
                return steps_done, raw_rewards, losses_tensor

    def select_action(self, steps_done, policy_net, _state,
                      eps_start, eps_end, eps_decay):
        sample = random.random()
        eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * steps_done / eps_decay)

        if sample > eps_threshold:
            # perform action using policy
            with (torch.no_grad()):
                return (self.full_action_space[policy_net(_state).argmax()],
                        policy_net(_state).argmax().view(1, 1))

        else:
            # perform random action
            random_choice = random.randint(0, len(self.full_action_space) - 1)
            return (self.full_action_space[random_choice],
                    torch.tensor(random_choice, device=self.device, dtype=torch.int).view(1, 1))

    def optimize_model(self, policy_net: torch.nn.Module, target_net: torch.nn.Module):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                      device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device, dtype=torch.float)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        criterion = torch.nn.MSELoss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()

        # inplace gradient clipping
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=5.0)
        self.optimizer.step()

        return loss

    def get_full_action_space(self):
        all_actions = []

        for elem_1 in self.local_action_space:
            action_1 = [elem_1]
            for elem_2 in self.local_action_space:
                action_2 = action_1 + [elem_2]
                for elem_3 in self.local_action_space:
                    current_action = action_2 + [elem_3]
                    all_actions.append(torch.tensor(current_action, device=self.device, dtype=torch.float))

        return all_actions
