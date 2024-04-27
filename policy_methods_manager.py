import os
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

import utils
import models

import sys

sys.path.append(r'E:\Alex\UniBuc\MasterThesis\gym-satellite-ca')

from gym_satellite_ca.envs import satDataClass

# constant paths
BASE = "./"
DATA_PATH = os.path.join(BASE, "data")

# paths
sat_data_config = os.path.join(DATA_PATH, "default_sat_data_config.json")

# setting up the satellite data and init config of the environment
init_sat = utils.get_sat_data_env(sat_data_config)

# setting up the environment
env = gym.make('gym_satellite_ca:gym_satellite_ca/CollisionAvoidance-v0',
               satellite=init_sat)

# set up neural net configuration
nn_conf = {
    "init_layer": 4,
    "hidden_layer": 16,
    "output_layer": 6
}

# instantiate the policy
nn_policy = models.NN_NeuralNet.SatColAvoidPolicy(conf=nn_conf)

# instantiate the utilities class
nn_utils = models.NU_NeuralUtils.NeuralNetUtils(game_env=env)
optimizer = nn_utils.instantiate_loss_fnc_optimiser(policy=nn_policy)

train_eval_steps = 100
train_total_losses, train_total_iters = [], []
eval_total_iters = []

for steps in range(train_eval_steps):
    # train the model for n times
    train_losses, train_iters = nn_utils.train_policy(policy=nn_policy, game_env=env,
                                                      optimizer=optimizer, num_train_iterations=5)

    # eval the model for n times
    eval_iters = nn_utils.eval_policy(policy=nn_policy, game_env=env,
                                      optimizer=optimizer, num_eval_iterations=1)

    print(f"Epoch {steps} - Train Iters Mean: {np.mean(train_iters)} | Eval Iters Mean: {np.mean(eval_iters)}")

    train_total_losses.extend(train_losses)
    train_total_iters.append(np.mean(train_iters))
    eval_total_iters.append(np.mean(eval_iters))

n_loss = np.arange(0, len(train_total_losses))
n_iters = np.arange(0, len(train_total_iters))

plt.plot(n_loss, train_total_losses)
plt.title(f"Losses")
plt.xlabel(f"Games played")
plt.ylabel(f"Loss value")
plt.grid()
plt.show()

plt.plot(n_iters, train_total_iters)
plt.plot(n_iters, eval_total_iters)
plt.title(f"Iterations Played")
plt.xlabel(f"Games played")
plt.ylabel(f"Total num Iters")
plt.grid()
plt.show()

