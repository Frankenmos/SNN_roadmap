import numpy as np
from pysc2.agents import base_agent

# ---- PySC2 colors.py fix for Python 3.10+ ----
import random
from pysc2.lib import colors as _colors

def _shuffled_hue_fixed(scale):
    # Same idea as the GitHub issue workaround
    palette = list(_colors.smooth_hue_palette(scale))
    random_keys = [random.random() for _ in palette]
    palette = [x for _, x in sorted(zip(random_keys, palette))]
    return np.array(palette)

_colors.shuffled_hue = _shuffled_hue_fixed
# ----------------------------------------------

from pysc2.lib import actions, features
import torch

from PPO_CNN.policy_network import PolicyNetwork
from PPO_CNN.PPO import PPO
from obs_space.obs_space_2 import ObservationExtractor
from action_space.action_space import ActionSpace
from PPO_CNN.reward_function_2 import RewardFunctionV2Half
from Utility.config import cfg

_PLAYER_FRIENDLY = 1
_PLAYER_ENEMY = 4


class DefeatRoaches(base_agent.BaseAgent):
    def __init__(
        self,
        spatial_input_shape=None,
        vector_input_dim=None,
        action_dim=None,
        lr=None,
        gamma=None,
        clip_eps=None,
        snn_state=None,
    ):
        super(DefeatRoaches, self).__init__()
        self.steps = 0
        self.extractor = ObservationExtractor()
        self.action_space = ActionSpace()
        self.reward_function = RewardFunctionV2Half()
        self.snn_state = snn_state

        if spatial_input_shape is None:
            spatial_input_shape = tuple(cfg.model.spatial_input_shape)
        if vector_input_dim is None:
            vector_input_dim = cfg.model.vector_input_dim
        if action_dim is None:
            action_dim = cfg.model.action_dim

        assert spatial_input_shape == tuple(
            cfg.model.spatial_input_shape
        ), f"Invalid spatial_input_shape: {spatial_input_shape}"
        assert vector_input_dim == cfg.model.vector_input_dim
        assert action_dim == cfg.model.action_dim

        self.policy = PolicyNetwork(spatial_input_shape, vector_input_dim, action_dim)
        self.policy.to(self.policy.device)  # make sure it's really on GPU

        self.ppo = PPO(
            policy_net=self.policy,
            lr=lr if lr is not None else cfg.hyperparameters.lr,
            gamma=gamma if gamma is not None else cfg.hyperparameters.gamma,
            clip_epsilon=clip_eps if clip_eps is not None else cfg.hyperparameters.clip_eps,
        )

        self.selected_armies = []

    def step(self, obs):
        """Compatibility wrapper for the environment.

        The BaseAgent.step signature expects a FunctionCall to be returned.
        We keep that contract here and expose the richer training tuple via
        `step_with_data` for training loops and tests that need extra values.
        """
        super(DefeatRoaches, self).step(obs)
        # Use the data-collecting helper, but return only the FunctionCall
        action_func, *_rest = self.step_with_data(obs)
        return action_func

    def step_with_data(self, obs):
        """Perform a step and return data useful for training/recording.

        Returns a tuple:
            (action_func, action_id, log_prob, value, spatial_obs, vector_obs, reward, xy_raw)
        """
        # Maintain step counter and call extractor
        self.steps += 1

        spatial_observation, vector_observation = self.extractor.extract_observation(obs)

        # Policy now returns xy instead of angle
        # select_action returns TENSORS now: action_id, xy_env, log_prob_total, value, next_state, xy_raw_sample
        action_id_tensor, xy_env_tensor, log_prob_tensor, value_tensor, self.snn_state, xy_raw_tensor = self.ppo.select_action(
            (spatial_observation, vector_observation), state=self.snn_state
        )
        
        # For Action Space (needs CPU/Numpy)
        xy_cpu = xy_env_tensor.squeeze(0).cpu().numpy()
        action_id_int = int(action_id_tensor.item())
        
        # CRITICAL FIX: Always keep army selected
        action_func = actions.FUNCTIONS.no_op()
        can_select_army = actions.FUNCTIONS.select_army.id in obs.observation.available_actions
        
        if self.steps < 2 and can_select_army:
            action_func = actions.FUNCTIONS.select_army("select")
        else:
            if action_id_int == 0: # Attack
                action_func = self.action_space.attack(obs, xy_cpu)
            elif action_id_int == 1: # Move
                action_func = self.action_space.move(obs, xy_cpu)
            else:
                action_func = actions.FUNCTIONS.no_op()
                
        # Reward
        reward = self.reward_function.calculate_reward(obs, vector_observation)
        # Reward is scalar float from reward_function
        
        # Return TENSORS for training data, Scalars for pysc2/logging where needed
        # (action_func, action_id_tensor, log_prob_tensor, value_tensor, spatial, vector, reward_scalar, xy_raw_tensor)
        # Updated to include xy_raw_tensor for PPO training
        return action_func, action_id_tensor, log_prob_tensor, value_tensor, spatial_observation, vector_observation, reward, xy_raw_tensor

    def reset(self):
        super(DefeatRoaches, self).reset()
        self.snn_state = None
        self.extractor.reset()
        self.reward_function.reset()
        self.selected_armies = []

    def update_policy(self):
        self.ppo.update_policy(
            batch_size=cfg.hyperparameters.batch_size,
            epochs=cfg.hyperparameters.epochs,
        )
