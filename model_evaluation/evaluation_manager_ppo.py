import os
import torch
import datetime

import ppo_evaluator

import sys

sys.path.append(r'E:\Alex\UniBuc\MasterThesis\gym-satellite-ca')

from gym_satellite_ca.envs import satDataClass

# set device
device = torch.device('cpu')

# constant paths
BASE = r"E:\Alex\UniBuc\MasterThesis\src"
DATA_PATH = os.path.join(BASE, "data")
MODEL_DIR_PATH = os.path.join(BASE, "recorded_models", "ppo_models_kepl", "ppo_model_16", "ppo_model_16_s0")
MODEL_FILE_PATH = os.path.join(MODEL_DIR_PATH, "ppo_model")
sat_data_config = os.path.join(DATA_PATH, "default_sat_data_config.json")

print(f"{datetime.datetime.now()} - Evaluation started!")
evaluator = ppo_evaluator.PPOEvaluator(
    device=device,
    model_dir_path=MODEL_DIR_PATH,
    model_file_path=MODEL_FILE_PATH,
    model_evaluation_path=MODEL_DIR_PATH,
    sat_data_config=sat_data_config)

evaluator.perform_evaluation(game_env=evaluator.game_env,
                             policy=evaluator.policy,
                             num_runs=30)

print(f"{datetime.datetime.now()} - Evaluation finished!")
