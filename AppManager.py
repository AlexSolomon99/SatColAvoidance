import gymnasium
import sys

import numpy as np

import AppUtils

sys.path.append(r'E:\Alex\UniBuc\MasterThesis\gym-satellite-ca')

env = gymnasium.make('gym_satellite_ca:gym_satellite_ca/CollisionAvoidance-v0')
print(env.observation_space)
print(env.action_space.shape[0])

obs, _ = env.reset()
print("TCA Time Lapse shape", obs['tca_time_lapse'].shape)
print("Primary mass", obs['primary_sc_mass'])
print("Primary Sc Seq shape", obs['primary_sc_state_seq'].shape)
print("Secondary Seq shape", obs['secondary_sc_state_seq'].shape)

# print(np.min(obs['primary_sc_state_seq']))
# print(np.max(obs['primary_sc_state_seq']))
# print(np.min(obs['secondary_sc_state_seq']))
# print(np.max(obs['secondary_sc_state_seq']))

rewards, observations, actions = AppUtils.play_game_manually(game_env=env)
