import os
import torch
import datetime

import dqn_evaluator

import sys

sys.path.append(r'E:\Alex\UniBuc\MasterThesis\gym-satellite-ca')

from gym_satellite_ca.envs import satDataClass

# set device
device = torch.device('cuda')

# constant paths
BASE = r"E:\Alex\UniBuc\MasterThesis\src"
DATA_PATH = os.path.join(BASE, "data")
for idx in range(1):
    MODEL_DIR_PATH = os.path.join(BASE, "recorded_models", "dqn_models_kepl", "dqn_model_super_nice_dir")
    MODEL_FILE_PATH = os.path.join(MODEL_DIR_PATH, f"dqn_model_6")
    # MODEL_EVALUATION_PATH = os.path.join(MODEL_DIR_PATH, f"dqn_model_10_{idx+1}_dir")
    # if not os.path.isdir(MODEL_EVALUATION_PATH):
    #     os.mkdir(MODEL_EVALUATION_PATH)
    sat_data_config = os.path.join(DATA_PATH, "default_sat_data_config.json")

    # set the reset options
    reset_options = {
        "propagator": "numerical",
        "generate_sat": True
    }

    print(f"{datetime.datetime.now()} - Evaluation started!")
    evaluator = dqn_evaluator.DQNEvaluator(
        device=device,
        model_dir_path=MODEL_DIR_PATH,
        model_file_path=MODEL_FILE_PATH,
        model_evaluation_path=MODEL_DIR_PATH,
        sat_data_config=sat_data_config)

    evaluator.perform_evaluation(game_env=evaluator.game_env,
                                 policy=evaluator.policy,
                                 num_runs=30,
                                 reset_options=reset_options)

    print(f"{datetime.datetime.now()} - Evaluation finished!")
