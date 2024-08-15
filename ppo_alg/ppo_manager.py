import os
import gymnasium as gym

from ppo_alg.ppo import ppo
from ppo_alg.ppo_utils import setup_logger_kwargs
from ppo_alg.mpi_tools import mpi_fork
import utils
import ppo_alg.core as core

# constant paths
BASE = r"E:\Alex\UniBuc\MasterThesis\src"
DATA_PATH = os.path.join(BASE, "data")

# environment setup
# sat_data_config = os.path.join(DATA_PATH, "default_sat_data_config.json")
# env, data_preprocessing = utils.set_up_environment(sat_data_config=sat_data_config)
env = gym.make("CartPole-v1", max_episode_steps=100)

# ppo_alg algorithm parameters
num_cpu = 1
seed = 0
steps_per_epoch = 4000
epochs = 10
gamma = 0.99

# set up the ppo_alg algorithm
mpi_fork(num_cpu)  # run parallel code with mpi

logger_kwargs = setup_logger_kwargs(exp_name="ppo_try_cartpole",
                                    seed=seed,
                                    data_dir=DATA_PATH)

# run the ppo_alg algorithm
ppo(env=env, actor_critic=core.MLPActorCritic, gamma=gamma,
    seed=seed, steps_per_epoch=steps_per_epoch, epochs=epochs,
    logger_kwargs=logger_kwargs)
