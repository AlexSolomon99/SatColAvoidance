import torch
import numpy as np

from policy_methods_utils import PolicyMethodsUtils
import dataprocessing


class NeuralNetUtils:

    def __init__(self, game_env, game_utils: PolicyMethodsUtils,
                 observation_processing: dataprocessing.data_processing.ObservationProcessing):
        self.game_env = game_env
        self.game_utils = game_utils
        self.observation_processing = observation_processing


