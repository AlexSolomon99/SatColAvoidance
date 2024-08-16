import os
import gymnasium as gym

from ppo_alg.ppo import ppo
from ppo_alg.ppo_utils import setup_logger_kwargs
from ppo_alg.mpi_tools import mpi_fork
import utils
import ppo_alg.core as core

# ppo_alg algorithm parameters
num_cpu = 1
seed = 0
# 5 games per epoch
steps_per_epoch = 2
epochs = 2
gamma = 0.99
MODEL_NAME = "ppo_model"
obs_dim = 56
buffer_max_size = steps_per_epoch * 1151

# constant paths
BASE = r"E:\Alex\UniBuc\MasterThesis\src"
DATA_PATH = os.path.join(BASE, "data")
RECORDED_MODELS_PATH = os.path.join(BASE, "recorded_models")
PPO_METHODS_MODELS = os.path.join(RECORDED_MODELS_PATH, "ppo_models")

if not os.path.isdir(RECORDED_MODELS_PATH):
    os.mkdir(RECORDED_MODELS_PATH)

if not os.path.isdir(PPO_METHODS_MODELS):
    os.mkdir(PPO_METHODS_MODELS)

# paths - set up the model record dictionary
ppo_record_dict_path = os.path.join(PPO_METHODS_MODELS, "dqn_record.json")

if not os.path.isfile(ppo_record_dict_path):
    model_record_dict = {}
    model_record_last_idx = 0
else:
    model_record_dict = utils.read_json(json_path=ppo_record_dict_path)
    model_record_last_idx = len(model_record_dict.keys())
experiment_name = f"{MODEL_NAME}_{model_record_last_idx + 1}"

# environment setup
sat_data_config = os.path.join(DATA_PATH, "default_sat_data_config.json")
env, data_preprocessing = utils.set_up_environment(sat_data_config=sat_data_config)

# set up the ppo_alg algorithm
mpi_fork(num_cpu)  # run parallel code with mpi

logger_kwargs = setup_logger_kwargs(exp_name=experiment_name,
                                    seed=seed,
                                    data_dir=PPO_METHODS_MODELS)

# run the ppo_alg algorithm
ppo(env=env, data_preprocessing=data_preprocessing, obs_dim=obs_dim, actor_critic=core.MLPActorCritic, gamma=gamma,
    seed=seed, steps_per_epoch=steps_per_epoch, epochs=epochs, buffer_max_size=buffer_max_size,
    logger_kwargs=logger_kwargs, save_freq=1, model_record_dict=model_record_dict,
    model_record_last_idx=model_record_last_idx, record_dict_path=ppo_record_dict_path)
