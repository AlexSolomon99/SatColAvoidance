import torch
import random
import math
import copy
from itertools import count

from dqn_replay_memory import Transition


class DQNUtils:

    def __init__(self, observation_processing, memory, optimizer, device, eps_start: float, eps_end: float, eps_decay: float,
                 tau: float, gamma: float, batch_size: int):
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

    def play_game_once(self, game_env, policy_net: torch.nn.Module, target_net: torch.nn.Module,
                       steps_done: int):

        # initialise the reward tensor
        raw_rewards = torch.tensor([], device=self.device)

        # initialize the environment and get its state
        obs, _ = game_env.reset()
        # transform the observations
        flat_state = self.observation_processing.transform_observations(game_env_obs=obs)
        state = torch.from_numpy(flat_state).to(device=self.device, dtype=torch.float)

        for t in count():
            steps_done += 1

            # select an action
            action = self.select_action(game_env=game_env, steps_done=steps_done, policy_net=policy_net,
                                        _state=state,
                                        eps_start=self.eps_start, eps_end=self.eps_end, eps_decay=self.eps_decay,
                                        device=self.device)

            # apply the action and get the reward and next state
            print(f"Action: {action}")
            obs, reward, done, truncated, info = game_env.step(action.tolist())
            reward = torch.tensor([reward], device=self.device)
            done = done or truncated

            raw_rewards = torch.cat((raw_rewards, reward))

            if done:
                next_state = None
            else:
                next_flat_state = self.observation_processing.transform_observations(game_env_obs=obs)
                next_state = torch.from_numpy(next_flat_state).to(device=self.device, dtype=torch.float)

            # push the transition to the memory
            self.memory.push(state.unsqueeze(0), action, next_state.unsqueeze(0), reward)
            state = copy.deepcopy(next_state)

            # optimise the model
            self.optimize_model(policy_net=policy_net, target_net=target_net)

            # soft update of the target network's weights
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = (policy_net_state_dict[key] * self.tau + target_net_state_dict[key] *
                                              (1 - self.tau))
            target_net.load_state_dict(target_net_state_dict)

            if done:
                return steps_done, raw_rewards

    def select_action(self, game_env, steps_done, policy_net, _state, device,
                      eps_start, eps_end, eps_decay):
        sample = random.random()
        eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1.*steps_done / eps_decay)

        if sample > eps_threshold:
            # perform action using policy
            with torch.no_grad():
                print(f"Chose through policy!")
                return policy_net(_state)

        else:
            # perform random action
            print(f"Chose randomly!")
            return torch.tensor(game_env.action_space.sample(), device=device, dtype=torch.float)

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
        # print(f"State Batch: {state_batch}")
        # print(f"Action Batch: {action_batch}")
        # print(f"Reward batch: {reward_batch}")
        # print()

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        draft_state_action = policy_net(state_batch)
        state_action_values = policy_net(state_batch).gather(1, action_batch)
        # print(f"DRAFT STATE ACTION: {draft_state_action}")
        # print(f"STATE ACTION VALUE: {state_action_values}")
        # print()

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states)
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()

        # inplace gradient clipping
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        self.optimizer.step()
