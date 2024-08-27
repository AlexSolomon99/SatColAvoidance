import os
import gymnasium as gym

from ppo_alg.ppo import ppo
from ppo_alg.ppo_utils import setup_logger_kwargs
from ppo_alg.mpi_tools import mpi_fork
import utils
import ppo_alg.core as core

# ppo_alg algorithm parameterss
seed = 0
# 5 games per epoch
steps_per_epoch = 5 * 1151
# steps_per_epoch = 5000
epochs = 100
gamma = 0.99
MODEL_NAME = "ppo_model"
obs_dim = 9
PI_LR = 3e-4
LR_VF = 1e-3

# constant paths
BASE = r"E:\Alex\UniBuc\MasterThesis\src"
DATA_PATH = os.path.join(BASE, "data")
RECORDED_MODELS_PATH = os.path.join(BASE, "recorded_models")
PPO_METHODS_MODELS = os.path.join(RECORDED_MODELS_PATH, "ppo_models_kepl")
# PPO_METHODS_MODELS = os.path.join(RECORDED_MODELS_PATH, "cartpole")

if not os.path.isdir(RECORDED_MODELS_PATH):
    os.mkdir(RECORDED_MODELS_PATH)

if not os.path.isdir(PPO_METHODS_MODELS):
    os.mkdir(PPO_METHODS_MODELS)

# paths - set up the model record dictionary
ppo_record_dict_path = os.path.join(PPO_METHODS_MODELS, "ppo_record.json")

if not os.path.isfile(ppo_record_dict_path):
    model_record_dict = {}
    model_record_last_idx = 0
else:
    model_record_dict = utils.read_json(json_path=ppo_record_dict_path)
    model_record_last_idx = len(model_record_dict.keys())
experiment_name = f"{MODEL_NAME}_{model_record_last_idx + 1}"

# environment setup
sat_data_config = os.path.join(DATA_PATH, "default_sat_data_config.json")
env = utils.set_up_kepl_environment(sat_data_config=sat_data_config)
# env = gym.make("CartPole-v1", max_episode_steps=1000)

# set up the ppo_alg algorithm
logger_kwargs = setup_logger_kwargs(exp_name=experiment_name,
                                    seed=seed,
                                    data_dir=PPO_METHODS_MODELS)

# run the ppo_alg algorithm
ppo(env=env, obs_dim=obs_dim, actor_critic=core.MLPActorCritic, gamma=gamma,
    seed=seed, steps_per_epoch=steps_per_epoch, epochs=epochs, pi_lr=PI_LR, vf_lr=LR_VF,
    logger_kwargs=logger_kwargs, save_freq=1, model_record_dict=model_record_dict,
    model_record_last_idx=model_record_last_idx, record_dict_path=ppo_record_dict_path)
