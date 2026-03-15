import json
import logging
import os
import time

import numpy as np

from .sram import ReflexSRAM

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces


class ReflexEnv(gym.Env):
    def __init__(self):
        super(ReflexEnv, self).__init__()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Dict({
            'stress': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            'risk': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            'friction': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            'success_ratio': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        })
        self.state = {}

    def _get_obs(self):
        return {
            'stress': np.array([float(self.state.get('stress', 0.0))], dtype=np.float32),
            'risk': np.array([float(self.state.get('risk', 0.0))], dtype=np.float32),
            'friction': np.array([float(self.state.get('friction', 0.0))], dtype=np.float32),
            'success_ratio': np.array([float(self.state.get('success_ratio', 0.0))], dtype=np.float32),
        }

    def step(self, action):
        return self._get_obs(), 0.0, False, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return self._get_obs(), {}


class ReflexBrain:
    """Consumes manifest-derived tool metrics/costs and emits soft regulation bias."""

    def __init__(self, iface='tool0', sram_path=None):
        self.iface = iface
        self.env = ReflexEnv()
        self.state = {}

        self.ticker_period = 10.0
        self.ticker_min = 1.0
        self.ticker_max = 30.0

        self._cmd_latency = 0.0
        self._inj_rate = 0.0
        self._last_save = 0.0

        self._baseline = {
            'safe_interaction_scale': 1.0,
            'safe_disruption_cooldown': 0.0,
            'safe_interaction_cooldown': 0.0,
            'last_stable_pressure': 0.0,
        }
        self._current_disruption_cooldown = 0.0
        self._current_interaction_cooldown = 0.0
        self._current_interaction_scale = 1.0

        self.sram = ReflexSRAM(path=(sram_path or '/root/brains/default/reflex.pkl'))
        self._load_state()

        self._phase = 'NEONATAL'
        self._phase_ts = time.time()
        self._stable_ticks = 0

    def _load_state(self):
        data = self.sram.load()
        if not data and os.path.exists('/root/.reflex_state.json'):
            try:
                with open('/root/.reflex_state.json', 'r') as f:
                    data = json.load(f)
                logging.info('[Reflex] Migrated legacy state to SRAM.')
            except Exception:
                pass

        if data:
            decay = 0.95
            self._baseline['safe_disruption_cooldown'] = data.get('safe_disruption_cooldown', 0.0) * decay
            self._baseline['safe_interaction_cooldown'] = data.get('safe_interaction_cooldown', 0.0) * decay
            saved_scale = data.get('safe_interaction_scale', 1.0)
            self._baseline['safe_interaction_scale'] = 1.0 - (1.0 - saved_scale) * decay
            self._baseline['last_stable_pressure'] = data.get('last_stable_pressure', 0.0)

    def _maybe_persist(self):
        now = time.time()
        if now - self._last_save < 300:
            return

        if self.stress_level < 0.2:
            self._baseline['safe_interaction_scale'] = min(self._baseline['safe_interaction_scale'], self._current_interaction_scale)
            self._baseline['safe_disruption_cooldown'] = max(self._baseline['safe_disruption_cooldown'], self._current_disruption_cooldown)
            self._baseline['safe_interaction_cooldown'] = max(self._baseline['safe_interaction_cooldown'], self._current_interaction_cooldown)
            self._baseline['last_stable_pressure'] = self.stress_level

            data = {
                'safe_interaction_scale': self._baseline['safe_interaction_scale'],
                'safe_disruption_cooldown': self._baseline['safe_disruption_cooldown'],
                'safe_interaction_cooldown': self._baseline['safe_interaction_cooldown'],
                'last_stable_pressure': self._baseline['last_stable_pressure'],
                'hardware_signature': f'{self.iface}',
                'last_update': int(now),
            }
            self.sram.save(data)
            self._last_save = now

    def _body_ready(self):
        return self.stress_level < 0.2 and self.risk < 0.2

    def observe(self, observation):
        tool_metrics = observation.get('tool_metrics', {}) if isinstance(observation, dict) else {}
        tool_costs = observation.get('tool_costs', {}) if isinstance(observation, dict) else {}

        command_latency = float(tool_metrics.get('command_latency', 0.0))
        alpha = 0.2
        self._cmd_latency = (1 - alpha) * self._cmd_latency + alpha * command_latency

        self._inj_rate = float(tool_metrics.get('injection_pressure', 0.0))
        self.regulate_ticker(self._inj_rate)

        self.state = {
            'tool_metrics': dict(tool_metrics),
            'tool_costs': dict(tool_costs),
            'success_ratio': float(observation.get('success_ratio', 0.0)),
            'failure_ratio': float(observation.get('failure_ratio', 0.0)),
            'error_ratio': float(observation.get('error_ratio', 0.0)),
            'no_targets_for': float(observation.get('no_targets_for', 0.0)),
            'active_for': float(observation.get('active_for', 0.0)),
            'inactive_for': float(observation.get('inactive_for', 0.0)),
            'cpu_load': float(observation.get('cpu_load', 0.0)),
            'mem_usage': float(observation.get('mem_usage', 0.0)),
            'temperature': float(observation.get('temperature', 0.0)),
        }
        self.env.state = {
            'stress': self.stress_level,
            'risk': self.risk,
            'friction': self.friction,
            'success_ratio': self.state['success_ratio'],
        }

        now = time.time()
        phase_duration = now - self._phase_ts
        if self._phase == 'NEONATAL':
            self._stable_ticks = self._stable_ticks + 1 if self._body_ready() else 0
            if self._stable_ticks >= 5 or phase_duration > 120:
                self._phase = 'CALIBRATING'
                self._phase_ts = now
                self._stable_ticks = 0
        elif self._phase == 'CALIBRATING':
            self._stable_ticks = self._stable_ticks + 1 if self._body_ready() else 0
            if self._stable_ticks >= 10 or phase_duration > 120:
                self._phase = 'OPERATIONAL'
                self._phase_ts = now

    def regulate_ticker(self, pressure=0.0):
        pressure = max(0.0, min(float(pressure), 1.0))
        target = self.ticker_min + pressure * (self.ticker_max - self.ticker_min)
        if target > self.ticker_period:
            self.ticker_period += (target - self.ticker_period) * 0.4
        else:
            self.ticker_period += (target - self.ticker_period) * 0.05
        self.ticker_period = max(self.ticker_period, self.ticker_min)

    def bias(self):
        modifiers = {
            'recon_time_multiplier': 1.0,
            'interaction_scale': 1.0,
            'ticker_period': self.ticker_period,
            'disruption_cooldown': 0.0,
            'interaction_cooldown': 0.0,
        }

        if self._phase == 'NEONATAL':
            return {
                'recon_time_multiplier': 2.5,
                'interaction_scale': 0.15,
                'ticker_period': max(self.ticker_period, self.ticker_max),
                'disruption_cooldown': 5.0,
                'interaction_cooldown': 2.0,
            }

        if self._phase == 'CALIBRATING':
            return {
                'recon_time_multiplier': 1.5,
                'interaction_scale': 0.4,
                'ticker_period': self.ticker_period,
                'disruption_cooldown': 2.0,
                'interaction_cooldown': 1.0,
            }

        stress = self.stress_level
        risk = self.risk
        friction = self.friction

        if (stress > 0.8 and self.ticker_period >= self.ticker_max) or risk > 0.8:
            modifiers['recon_time_multiplier'] = 0.5
            modifiers['interaction_scale'] = 0.2
        elif stress > 0.5:
            modifiers['recon_time_multiplier'] = 2.0
            modifiers['interaction_scale'] = 0.5

        if stress < 0.3 and self.state.get('success_ratio', 0.0) > 0.0:
            modifiers['interaction_scale'] = 1.5
            modifiers['recon_time_multiplier'] = 0.8

        modifiers['interaction_scale'] = max(modifiers['interaction_scale'], 0.2)
        modifiers['interaction_scale'] *= 1.0 + 0.5 * friction
        modifiers['recon_time_multiplier'] *= 1.0 + 0.2 * friction

        pressure = min(1.0, (self._inj_rate * 2.0) + (self._cmd_latency * 2.0))
        if pressure > 0.1:
            modifiers['disruption_cooldown'] = pressure * 2.5
            modifiers['interaction_cooldown'] = pressure * 1.0

        modifiers['disruption_cooldown'] = max(modifiers['disruption_cooldown'], self._baseline['safe_disruption_cooldown'])
        modifiers['interaction_cooldown'] = max(modifiers['interaction_cooldown'], self._baseline['safe_interaction_cooldown'])
        modifiers['interaction_scale'] = min(modifiers['interaction_scale'], self._baseline['safe_interaction_scale'])

        self._current_disruption_cooldown = modifiers['disruption_cooldown']
        self._current_interaction_cooldown = modifiers['interaction_cooldown']
        self._current_interaction_scale = modifiers['interaction_scale']

        self._maybe_persist()
        return modifiers

    @property
    def stress_level(self):
        costs = self.state.get('tool_costs', {})
        cost_stress = float(costs.get('stress', 0.0))

        metrics = self.state.get('tool_metrics', {})

        # Derived pressures from system state
        cpu_p = self.state.get('cpu_load', 0.0)
        temp = self.state.get('temperature', 0.0)
        thermal_p = 0.0
        if temp > 50.0:
            thermal_p = min(1.0, (temp - 50.0) / 35.0)

        derived = min(1.0, max(
            float(metrics.get('io_pressure', 0.0)),
            float(metrics.get('injection_pressure', 0.0)),
            float(metrics.get('thermal_pressure', 0.0)),
            float(metrics.get('latency_pressure', 0.0)),
            float(metrics.get('driver_instability', 0.0)),
            float(self.state.get('error_ratio', 0.0)),
            cpu_p,
            thermal_p,
        ))
        return min(1.0, max(cost_stress, derived))

    @property
    def risk(self):
        costs = self.state.get('tool_costs', {})
        cost_risk = float(costs.get('risk', 0.0))

        metrics = self.state.get('tool_metrics', {})
        derived = min(1.0, max(
            float(metrics.get('timeout_pressure', 0.0)),
            float(metrics.get('blindness', 0.0)),
            float(metrics.get('detection_risk', 0.0)),
            float(self.state.get('error_ratio', 0.0) * 0.8),
        ))
        return min(1.0, max(cost_risk, derived))

    @property
    def friction(self):
        costs = self.state.get('tool_costs', {})
        return min(1.0, max(0.0, float(costs.get('friction', 0.0))))

    @property
    def boredom(self):
        return self.friction
