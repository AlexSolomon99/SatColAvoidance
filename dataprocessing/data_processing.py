import sys
import copy

import numpy as np

sys.path.append(r'E:\Alex\UniBuc\MasterThesis\gym-satellite-ca')

from gym_satellite_ca.envs import satDataClass


class ObservationProcessing:

    # Constants Definition
    # Environment observation keys
    PRIMARY_CURRENT_PV = "primary_current_pv"
    SECONDARY_SC_STATE_SEQ = "secondary_sc_state_seq"
    TCA_TIME_LAPSE = "tca_time_lapse"
    PRIMARY_SC_MASS = "primary_sc_mass"

    def __init__(self, satellite_data: satDataClass.SatelliteData, tca_time_lapse_max_abs_val: float):
        self._satellite_data = satellite_data
        self._position_normaliser = copy.deepcopy(self.satellite_data.sma)
        self._velocity_normaliser = copy.deepcopy(self.satellite_data.sma * 1.0e-3)
        self._mass_normaliser = copy.deepcopy(self.satellite_data.mass)
        self._tca_time_lapse_max_abs_val = tca_time_lapse_max_abs_val
        self._obs_key_normaliser_dict = self.get_obskey_normaliser_dict()

    def transform_observations(self, game_env_obs: dict):
        # normalise the values of the observation
        normalised_obs_dict = self.normalise_observation_dict(game_env_obs=game_env_obs)

        # flatten the normalised observation dictionary
        flattened_data = self.flatten_observation_dict(observation_dict=normalised_obs_dict)

        return flattened_data

    def normalise_observation_dict(self, game_env_obs: dict):
        # normalise the values of the observation
        normalised_obs_dict = {}
        for key_ in self.obs_key_normaliser_dict:
            normalised_obs_dict[key_] = self.obs_key_normaliser_dict[key_](game_env_obs, key_)

        return normalised_obs_dict

    def flatten_observation_dict(self, observation_dict: dict):
        flattened_observation = []

        for key_ in self.obs_key_normaliser_dict:
            flattened_observation.append(observation_dict[key_].flatten())

        return np.concatenate(flattened_observation)

    def normalise_primary_current_pv(self, game_env_obs: dict, env_obs_key: str):
        """
        Method used to normalise the values of the PRIMARY_CURRENT_PV component of the observation space. The values of
        the component are positions and velocities of the satellite. They are normalised with the semi-major axis
        of the primary satellite's orbit.
        :param game_env_obs: The observation received from the game environment.
        :param env_obs_key: The key of the observation parameter.
        :return:
        """
        primary_current_pv = game_env_obs[env_obs_key].copy()

        primary_current_pos_norm = primary_current_pv[:3] / self.position_normaliser
        primary_current_vel_norm = primary_current_pv[3:] / self.velocity_normaliser

        return np.concatenate([primary_current_pos_norm, primary_current_vel_norm])

    def normalise_pv_state_seq(self, game_env_obs: dict, env_obs_key: str):
        """
        Method used to normalise the values of the PV state sequences components of the observation space. The values of
        the components  are positions and velocities of the satellite and they are normalised with the semi-major axis
        of the primary satellite's orbit.
        :param game_env_obs: The observation received from the game environment.
        :param env_obs_key: The key of the observation parameter.
        :return:
        """
        pv_state_sequence = copy.deepcopy(game_env_obs[env_obs_key])

        pv_state_sequence[:, :3] = pv_state_sequence[:, :3] / self.position_normaliser
        pv_state_sequence[:, 3:] = pv_state_sequence[:, 3:] / self.velocity_normaliser

        return pv_state_sequence

    def normalise_tca_time_lapse(self, game_env_obs: dict, env_obs_key: str):
        """
        Method used to normalise the values of the TCA time-lapse component of the observation space. The values of
        the component  are the number of seconds between the current time and the TCA of the event.
        :param game_env_obs: The observation received from the game environment.
        :param env_obs_key: The key of the observation parameter.
        :return:
        """
        tca_time_lapse = copy.deepcopy(game_env_obs[env_obs_key])

        return tca_time_lapse / self.tca_time_lapse_max_abs_val

    def normalise_sat_mass(self, game_env_obs: dict, env_obs_key: str):
        """
        Method used to normalise the values of the satellite mass component of the observation space. The values of
        the component are the number of kilograms of the satellite at the beginning of the event.
        :param game_env_obs: The observation received from the game environment.
        :param env_obs_key: The key of the observation parameter.
        :return:
        """
        satellite_mass = copy.deepcopy(game_env_obs[env_obs_key])

        return satellite_mass / self.mass_normaliser

    def get_obskey_normaliser_dict(self):
        return {
            self.PRIMARY_CURRENT_PV: self.normalise_primary_current_pv,
            self.SECONDARY_SC_STATE_SEQ: self.normalise_pv_state_seq,
            self.TCA_TIME_LAPSE: self.normalise_tca_time_lapse,
            self.PRIMARY_SC_MASS: self.normalise_sat_mass,
        }

    @property
    def satellite_data(self):
        return self._satellite_data

    @satellite_data.setter
    def satellite_data(self, x):
        self._satellite_data = x

    @property
    def position_normaliser(self):
        return self._position_normaliser

    @position_normaliser.setter
    def position_normaliser(self, x):
        self._position_normaliser = x

    @property
    def velocity_normaliser(self):
        return self._velocity_normaliser

    @velocity_normaliser.setter
    def velocity_normaliser(self, x):
        self._velocity_normaliser = x

    @property
    def mass_normaliser(self):
        return self._mass_normaliser

    @mass_normaliser.setter
    def mass_normaliser(self, x):
        self._mass_normaliser = x

    @property
    def tca_time_lapse_max_abs_val(self):
        return self._tca_time_lapse_max_abs_val

    @tca_time_lapse_max_abs_val.setter
    def tca_time_lapse_max_abs_val(self, x):
        self._tca_time_lapse_max_abs_val = x

    @property
    def obs_key_normaliser_dict(self):
        return self._obs_key_normaliser_dict

    @obs_key_normaliser_dict.setter
    def obs_key_normaliser_dict(self, x):
        self._obs_key_normaliser_dict = x
