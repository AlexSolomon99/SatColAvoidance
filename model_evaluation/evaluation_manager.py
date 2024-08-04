import os
import torch

import policy_evaluator

import sys

sys.path.append(r'E:\Alex\UniBuc\MasterThesis\gym-satellite-ca')

from gym_satellite_ca.envs import satDataClass

# set device
device = torch.device('cuda')

# constant paths
BASE = r"E:\Alex\UniBuc\MasterThesis\src"
DATA_PATH = os.path.join(BASE, "data")
MODEL_DIR_PATH = os.path.join(BASE, "recorded_models", "policy_methods_models", "policy_model_10_dir")
MODEL_FILE_PATH = os.path.join(MODEL_DIR_PATH, "policy_model_10")
sat_data_config = os.path.join(DATA_PATH, "default_sat_data_config.json")

evaluator = policy_evaluator.PolicyEvaluator(
    device=device,
    model_dir_path=MODEL_DIR_PATH,
    model_file_path=MODEL_FILE_PATH,
    model_evaluation_path=MODEL_DIR_PATH,
    sat_data_config=sat_data_config)
