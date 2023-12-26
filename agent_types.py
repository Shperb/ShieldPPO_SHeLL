from abc import ABC
import ppo_shieldLSTM

from enum import Enum


class ObservationType(Enum):
    Camera = 1
    Kinematics = 2


class Agent(ABC):
    def __init__(self, shield, obs_type, state_dim, action_dim, lr_actor, lr_critic, gamma, k_epochs, eps_clip,
                 has_continuous_action_space, action_std_init=0.6, masking_threshold=0, k_last_states=1, safety_threshold=0.5):
        self.ppo = ppo_shieldLSTM.ShieldPPO(shield, obs_type, state_dim, action_dim, lr_actor, lr_critic, gamma, k_epochs, eps_clip,
                                            has_continuous_action_space, action_std_init, masking_threshold=masking_threshold,
                                            k_last_states=k_last_states, safety_threshold=safety_threshold)


class CameraAgent(Agent):
    def __init__(self, shield, state_dim, action_dim, lr_actor, lr_critic, gamma, k_epochs, eps_clip,
                 has_continuous_action_space, action_std_init=0.6, masking_threshold=0, k_last_states=1, safety_threshold=0.5):
        super(CameraAgent, self).__init__(shield, ObservationType.Camera, state_dim, action_dim, lr_actor, lr_critic, gamma, k_epochs, eps_clip,
                                          has_continuous_action_space, action_std_init, masking_threshold=masking_threshold,
                                          k_last_states=k_last_states, safety_threshold=safety_threshold)


class KinematicsAgent(Agent):
    def __init__(self, shield, state_dim, action_dim, lr_actor, lr_critic, gamma, k_epochs, eps_clip,
                 has_continuous_action_space, action_std_init=0.6, masking_threshold=0, k_last_states=1, safety_threshold=0.5):
        super(KinematicsAgent, self).__init__(shield, ObservationType.Kinematics, state_dim, action_dim, lr_actor, lr_critic, gamma, k_epochs, eps_clip,
                                              has_continuous_action_space, action_std_init, masking_threshold=masking_threshold,
                                              k_last_states=k_last_states, safety_threshold=safety_threshold)
