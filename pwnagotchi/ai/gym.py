import logging
import os
import numpy as np
import gym as legacy_gym

try:
    import toml
except ImportError:
    toml = None

PRIMAL = False
try:
    config_path = '/etc/pwnagotchi/config.toml'
    if not os.path.exists(config_path):
        config_path = './config.toml'

    if toml and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = toml.load(f)
            if config.get('ai', {}).get('primal', False):
                PRIMAL = True
except Exception as e:
    logging.debug(f"[ai] error checking for primal mode: {e}")

if PRIMAL:
    try:
        import gymnasium as gym
        from gymnasium import spaces
        logging.info("[ai] primal mode enabled: using gymnasium")
    except ImportError:
        import gym
        from gym import spaces
        logging.warning("[ai] primal mode enabled but gymnasium not found, falling back to gym")
        PRIMAL = False
else:
    import gym
    from gym import spaces

import pwnagotchi.ai.featurizer as featurizer
import pwnagotchi.ai.reward as reward
from pwnagotchi.ai.parameter import Parameter
from pwnagotchi.ai.reflex import ReflexBrain


class Environment(gym.Env):
    metadata = {'render.modes': ['human']}
    params = [
        Parameter('min_rssi', min_value=-200, max_value=-50),
        Parameter('ap_ttl', min_value=30, max_value=600),
        Parameter('sta_ttl', min_value=60, max_value=300),

        Parameter('recon_time', min_value=5, max_value=60),
        Parameter('max_inactive_scale', min_value=3, max_value=10),
        Parameter('recon_inactive_multiplier', min_value=1, max_value=3),
        Parameter('hop_recon_time', min_value=5, max_value=60),
        Parameter('min_recon_time', min_value=1, max_value=30),
        Parameter('max_interactions', min_value=1, max_value=25),
        Parameter('max_misses_for_recon', min_value=3, max_value=10),
        Parameter('excited_num_epochs', min_value=5, max_value=30),
        Parameter('bored_num_epochs', min_value=5, max_value=30),
        Parameter('sad_num_epochs', min_value=5, max_value=30),
    ]

    def __init__(self, agent, epoch):
        super(Environment, self).__init__()
        self._agent = agent
        self._epoch = epoch
        self._epoch_num = 0
        iface = agent.config()['main']['iface'] if hasattr(agent, 'config') else 'mon0'
        self.reflex = None
        if hasattr(agent, '_reflex') and agent._reflex:
            self.reflex = agent._reflex
        elif hasattr(agent, 'config') and agent.config()['ai'].get('reflex', False):
            self.reflex = ReflexBrain(iface)

        self._last_render = None

        # see https://github.com/evilsocket/pwnagotchi/issues/583
        self._supported_channels = agent.supported_channels()
        self._extended_spectrum = any(ch > 170 for ch in self._supported_channels)
        self._histogram_size, self._observation_shape = featurizer.describe(self._extended_spectrum)

        Environment.params += [
            Parameter('_channel_%d' % ch, min_value=0, max_value=1, meta=ch + 1) for ch in
            range(self._histogram_size) if ch + 1 in self._supported_channels
        ]

        self.last = {
            'reward': 0.0,
            'observation': None,
            'policy': None,
            'params': {},
            'state': None,
            'state_v': None
        }

        self.action_space = legacy_gym.spaces.MultiDiscrete([p.space_size() for p in Environment.params if p.trainable])
        low = np.full(self._observation_shape, 0.0, dtype=np.float32)
        high = np.full(self._observation_shape, 1.0, dtype=np.float32)
        self.observation_space = legacy_gym.spaces.Box(low=low, high=high, shape=self._observation_shape, dtype=np.float32)
        self.reward_range = reward.range

    @staticmethod
    def policy_size():
        return len(list(p for p in Environment.params if p.trainable))

    @staticmethod
    def policy_to_params(policy):
        num = len(policy)
        params = {}

        assert len(Environment.params) == num

        channels = []

        for i in range(num):
            param = Environment.params[i]

            if '_channel' not in param.name:
                params[param.name] = param.to_param_value(policy[i])
            else:
                has_chan = param.to_param_value(policy[i])
                # print("%s policy:%s bool:%s" % (param.name, policy[i], has_chan))
                chan = param.meta
                if has_chan:
                    channels.append(chan)

        params['channels'] = channels

        return params

    def _next_epoch(self):
        logging.debug("[ai] waiting for epoch to finish ...")
        return self._epoch.wait_for_epoch_data()

    def _apply_policy(self, policy):
        new_params = Environment.policy_to_params(policy)
        self.last['policy'] = policy
        self.last['params'] = new_params
        self._agent.on_ai_policy(new_params)

    def _observe(self):
        """
        Encapsulates the current state of the environment.
        Synthesizes the 'Eco' (Environment) for the 'Genos' (Brain).
        """
        # This should return the 'state' dict that featurizer expects.
        return self.last.get('state', {})

    def step(self, policy):
        # Fetch the bias from the Reflex layer
        if self.reflex:
            reflex_bias = self.reflex.bias()
            self.last['reflex_bias'] = reflex_bias

            # Add reflex telemetry (physiology)
            self.last['physiology'] = {
                "iface_stress": self.reflex.stress_level,
                "blindbug_risk": self.reflex.risk,
            }
        else:
            self.last['reflex_bias'] = {}

        # create the parameters from the policy and update
        # update them in the algorithm
        self._apply_policy(policy)
        self._epoch_num += 1

        obs = self._observe()
        # Only drive reflex if we own it (it's not the agent's shared instance)
        if self.reflex and (not hasattr(self._agent, '_reflex') or self.reflex != self._agent._reflex):
            self.reflex.observe(obs)

        # wait for the algorithm to run with the new parameters
        state = self._next_epoch()

        # Merge reflex state into main state for logging/rendering
        if self.reflex and hasattr(self.reflex, 'state'):
            state.update({k: v for k, v in self.reflex.state.items() if k in ['timeout_errors', 'injection_errors', 'io_wait', 'is_promiscuous']})

        self.last['reward'] = state['reward']

        # Check if the reflex layer is signaling a 'Self-Sacrifice' or 'Rebirth'
        if self.reflex and self.last.get('physiology', {}).get('blindbug_risk', 0) > 0.9:
            logging.error("[ai] --- SYSTEM VIABILITY LOW: Reincarnation Likely ---")
            # You could add a negative reward here to teach the
            # main brain (Genos) to avoid the states leading to this crash.
            self.last['reward'] -= 5.0

        self.last['state'] = state
        self.last['state_v'] = featurizer.featurize(state, self._epoch_num)

        self._agent.on_ai_step()

        done = not self._agent.is_training()
        return self.last['state_v'], self.last['reward'], done, {}

    def reset(self, seed=None, options=None):
        # logging.info("[ai] resetting environment ...")
        if PRIMAL:
            super().reset(seed=seed)

        self._epoch_num = 0
        state = self._next_epoch()
        self.last['state'] = state
        self.last['state_v'] = featurizer.featurize(state, 1)

        return self.last['state_v']

    def _render_histogram(self, hist):
        for ch in range(self._histogram_size):
            if hist[ch]:
                logging.info("      CH %d: %s" % (ch + 1, hist[ch]))

    def render(self, mode='human', close=False, force=False):
        # when using a vectorialized environment, render gets called twice
        # avoid rendering the same data
        if self._last_render == self._epoch_num:
            return

        if not self._agent.is_training() and not force:
            return

        self._last_render = self._epoch_num

        logging.info("[ai] --- training epoch %d/%d ---" % (self._epoch_num, self._agent.training_epochs()))
        logging.info("[ai] REWARD: %f" % self.last['reward'])

        if 'physiology' in self.last:
            logging.info("[ai] PHYSIO: stress=%.2f risk=%.2f" % (
                self.last['physiology']['iface_stress'],
                self.last['physiology']['blindbug_risk']
            ))

        logging.debug("[ai] policy: %s" % ', '.join("%s:%s" % (name, value) for name, value in self.last['params'].items()))

        logging.info("[ai] observation:")
        for name, value in self.last['state'].items():
            if 'histogram' in name:
                logging.info("    %s" % name.replace('_histogram', ''))
                self._render_histogram(value)
            elif name in ['sad_for_epochs', 'bored_for_epochs', 'blind_for_epochs',
                          'inactive_for_epochs', 'active_for_epochs',
                          'timeout_errors', 'injection_errors', 'io_wait', 'is_promiscuous',
                          'temperature']:
                logging.info("    %s: %s" % (name, value))