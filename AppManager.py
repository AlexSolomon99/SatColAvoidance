import gymnasium
import sys
import datetime

import numpy as np

import utils

sys.path.append(r'E:\Alex\UniBuc\MasterThesis\gym-satellite-ca')

env = gymnasium.make('gym_satellite_ca:gym_satellite_ca/CollisionAvoidance-v0')

print(env.observation_space)
print(env.action_space)

obs, _ = env.reset()
print("TCA Time Lapse shape", obs['tca_time_lapse'].shape)
print("Primary mass", obs['primary_sc_mass'])
print("Primary Sc Seq shape", obs['primary_sc_state_seq'].shape)
print("Secondary Seq shape", obs['secondary_sc_state_seq'].shape)

# print(np.min(obs['primary_sc_state_seq']))
# print(np.max(obs['primary_sc_state_seq']))
# print(np.min(obs['secondary_sc_state_seq']))
# print(np.max(obs['secondary_sc_state_seq']))

print(f"Start game: {datetime.datetime.now()}")
for _ in range(2):
    rewards, observations, actions = AppUtils.play_constant_game_manually(game_env=env, constant_action=[0.0, 0.0, 0.0])
    print(f"Game finished")
    print(f"List of rewards: {rewards}")

    print(f"Sum of rewards: {np.sum(rewards)}")
    print()

