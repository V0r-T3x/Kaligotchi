import logging
import os
import time
import numpy as np

try:
    import toml
except ImportError:
    toml = None

PRIMAL = False
PRIMAL_REASON = "disabled"
try:
    config_path = '/etc/pwnagotchi/config.toml'
    if not os.path.exists(config_path):
        config_path = './config.toml'

    if toml and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = toml.load(f)
            if config.get('ai', {}).get('primal', False):
                PRIMAL = True
                PRIMAL_REASON = "config.ai.primal=true"
except Exception as e:
    logging.debug(f"[ai] error checking for primal mode: {e}")

if PRIMAL:
    try:
        import stable_baselines3
        sb3_version = getattr(stable_baselines3, '__version__', '0.0.0')
        if int(sb3_version.split('.')[0]) < 2:
            logging.debug("[ai] Primal mode enabled but Stable Baselines3 v%s (< 2.0.0) detected. Falling back to legacy gym.", sb3_version)
            PRIMAL = False
            PRIMAL_REASON = f"sb3<{2}"
    except Exception as e:
        logging.debug(f"[ai] error checking SB3 version for primal mode: {e}")

if PRIMAL:
    try:
        import gymnasium as gym
        from gymnasium import spaces
        logging.info("[ai] primal mode enabled: using gymnasium")
    except ImportError:
        import gym
        from gym import spaces
        logging.debug("[ai] primal mode enabled but gymnasium not found, falling back to gym")
        PRIMAL = False
        PRIMAL_REASON = "gymnasium_missing"
else:
    import gym
    from gym import spaces

import pwnagotchi
from pwnagotchi import utils
import pwnagotchi.kali.ai.featurizer as featurizer
import pwnagotchi.ui.faces as faces
import pwnagotchi.kali.ai.reward as reward
from pwnagotchi.kali.ai.reflex import ReflexBrain


DEFAULT_TOOL_ACTION = {'tool_id': 'none', 'command_id': 'noop'}


class Environment(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, agent, epoch):
        super(Environment, self).__init__()
        self._agent = agent
        self._epoch = epoch
        self._epoch_num = 0

        iface = agent.config()['main']['iface'] if hasattr(agent, 'config') else 'tool0'
        self.reflex = None
        if hasattr(agent, '_reflex') and agent._reflex:
            self.reflex = agent._reflex
        elif hasattr(agent, 'config') and agent.config()['ai'].get('reflex', False):
            self.reflex = ReflexBrain(iface)

        self._last_render = None
        self._last_tool_snapshot = {}
        self._histogram_size, self._observation_shape = featurizer.describe(False)

        self._tool_actions = agent.tool_actions() if hasattr(agent, 'tool_actions') else []

        self.last = {
            'reward': 0.0,
            'observation': None,
            'policy': None,
            'state': None,
            'state_v': None,
            'tool_action': dict(DEFAULT_TOOL_ACTION),
            'actions_executed': [],
            'reflex_bias': {},
            'physiology': {},
        }
        self._last_action_ts = {}

        action_size = max(1, len(self._tool_actions))
        self.action_space = spaces.Discrete(action_size)

        low = np.full(self._observation_shape, 0.0, dtype=np.float32)
        high = np.full(self._observation_shape, 1.0, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, shape=self._observation_shape, dtype=np.float32)
        self.reward_range = reward.range

    def set_actions(self, actions):
        self._tool_actions = list(actions or [])
        self.action_space = spaces.Discrete(max(1, len(self._tool_actions)))
        self.last['tool_action'] = dict(DEFAULT_TOOL_ACTION)
        self.last['actions_executed'] = []

    @staticmethod
    def policy_size():
        return 1

    def _epoch_runtime_budget(self, base_delay: float):
        cfg = self._agent.config() if hasattr(self._agent, 'config') else {}
        ai_cfg = cfg.get('ai', {}) if isinstance(cfg, dict) else {}
        kali_cfg = cfg.get('kali', {}) if isinstance(cfg, dict) else {}

        snapshot_targets = float(self._last_tool_snapshot.get('targets_visible', 0.0) or 0.0)
        state_targets = float((self.last.get('state') or {}).get('targets_visible', 0.0) or 0.0)
        targets_visible = max(snapshot_targets, state_targets)

        density_scale = min(max(targets_visible / 4.0, 0.0), 2.0)
        default_base = min(max(base_delay * 0.35, 8.0), 20.0)
        base_window = float(ai_cfg.get('epoch_window_base_sec', kali_cfg.get('epoch_window_base_sec', default_base)) or default_base)
        min_window = float(ai_cfg.get('epoch_window_min_sec', kali_cfg.get('epoch_window_min_sec', 8.0)) or 8.0)
        max_window = float(ai_cfg.get('epoch_window_max_sec', kali_cfg.get('epoch_window_max_sec', 30.0)) or 30.0)
        epoch_window = max(min_window, min(base_window + 4.0 * density_scale, max_window))

        base_budget = int(ai_cfg.get('epoch_action_budget', kali_cfg.get('epoch_action_budget', 2)) or 2)
        max_budget = int(ai_cfg.get('epoch_action_budget_max', kali_cfg.get('epoch_action_budget_max', 8)) or 8)
        action_budget = base_budget + int(round(density_scale * 2.0))
        action_budget = max(1, min(action_budget, max_budget))

        default_inter = max(0.2, min(epoch_window / max(action_budget, 1), 2.0))
        inter_action_delay = float(ai_cfg.get('epoch_inter_action_delay', kali_cfg.get('epoch_inter_action_delay', default_inter)) or default_inter)
        inter_action_delay = max(inter_action_delay, 0.0)

        bias = self.last.get('reflex_bias') or {}
        interaction_scale = float(bias.get('interaction_scale', 1.0) or 1.0)
        interaction_scale = max(0.2, min(interaction_scale, 2.0))
        action_budget = max(1, int(round(action_budget * interaction_scale)))
        inter_action_delay = max(inter_action_delay, float(bias.get('interaction_cooldown', 0.0) or 0.0))
        return epoch_window, action_budget, inter_action_delay

    def _tool_action_index_by_name(self, command_id: str):
        for idx, action in enumerate(self._tool_actions):
            if action.command_id == command_id:
                return idx
        return None

    def _choose_follow_up_action(self, snapshot: dict):
        targets = float((snapshot or {}).get('targets_visible', 0.0) or 0.0)
        last_actions = list(self.last.get('actions_executed') or [])
        last_cmd = last_actions[-1].split('.', 1)[1] if last_actions and '.' in last_actions[-1] else None
        now = time.time()
        bias = self.last.get('reflex_bias') or {}
        disruption_cd = float(bias.get('disruption_cooldown', 0.0) or 0.0)
        interaction_cd = float(bias.get('interaction_cooldown', 0.0) or 0.0)
        error_ratio = float((self.last.get('state') or {}).get('error_ratio', 0.0) or 0.0)
        if targets > 0:
            # Alternate engagement pressure instead of spamming assoc only.
            if last_cmd == 'interact_with_target':
                ordered = ['force_state_change', 'set_channel', 'focus_context', 'interact_with_target']
            elif last_cmd == 'force_state_change':
                ordered = ['interact_with_target', 'set_channel', 'focus_context', 'force_state_change']
            else:
                ordered = ['interact_with_target', 'force_state_change', 'set_channel', 'focus_context']
        else:
            ordered = ['explore_environment', 'focus_context', 'sync_events']

        for name in ordered:
            if name == 'force_state_change':
                if (now - float(self._last_action_ts.get(name, 0.0))) < disruption_cd:
                    continue
                if error_ratio >= 0.20:
                    continue
            if name == 'interact_with_target':
                if (now - float(self._last_action_ts.get(name, 0.0))) < interaction_cd:
                    continue
            idx = self._tool_action_index_by_name(name)
            if idx is not None:
                return idx
        return None

    def _execute_action_index(self, action_idx: int):
        if getattr(self._agent, '_ai_pause', False):
            return {
                'ok': False,
                'error': 'ai_paused',
                'signals': {},
                'metrics': {},
                'costs': {},
            }, None

        action_idx = max(0, min(int(action_idx), len(self._tool_actions) - 1))
        action = self._tool_actions[action_idx]
        logging.debug("[gym] executing tool action: %s.%s", action.tool_id, action.command_id)

        result = self._agent.execute_tool_action(action_idx, context={
            'epoch': self._epoch_num,
            'state': self.last.get('state', {}),
        })
        logging.debug("[gym] post-exec: got result for %s.%s", action.tool_id, action.command_id)

        logging.debug("[gym] post-exec: observe_toolbox begin")
        snapshot = self._agent.observe_toolbox()
        logging.debug("[gym] post-exec: observe_toolbox done")
        self._last_tool_snapshot = snapshot
        logging.debug("[gym] post-exec: epoch.observe begin")
        self._epoch.observe(snapshot)
        logging.debug("[gym] post-exec: epoch.observe done")
        logging.debug("[gym] post-exec: epoch.ingest_execution begin")
        self._epoch.ingest_execution(result)
        logging.debug("[gym] post-exec: epoch.ingest_execution done")

        self.last['tool_action'] = {'tool_id': action.tool_id, 'command_id': action.command_id}
        self._last_action_ts[action.command_id] = time.time()
        return result, action

    def _sanitize_policy_action(self, action_idx: int):
        action_idx = max(0, min(int(action_idx), len(self._tool_actions) - 1))
        action = self._tool_actions[action_idx]
        if action.command_id != 'force_state_change':
            return action_idx

        now = time.time()
        bias = self.last.get('reflex_bias') or {}
        disruption_cd = float(bias.get('disruption_cooldown', 0.0) or 0.0)
        error_ratio = float((self.last.get('state') or {}).get('error_ratio', 0.0) or 0.0)
        within_cd = (now - float(self._last_action_ts.get('force_state_change', 0.0))) < disruption_cd
        if within_cd or error_ratio >= 0.20:
            for fallback in ('interact_with_target', 'set_channel', 'focus_context'):
                idx = self._tool_action_index_by_name(fallback)
                if idx is not None:
                    return idx
        return action_idx

    def _run_tool_step(self, policy):
        logging.debug("[gym] _run_tool_step: policy=%s", policy)
        self._tool_actions = self._agent.tool_actions() if hasattr(self._agent, 'tool_actions') else []
        self.last['actions_executed'] = []
        self.last['ui_update'] = None

        runtime_state = getattr(self._agent, 'runtime_state', lambda: 'stopped')()
        if getattr(self._agent, '_ai_pause', False) or runtime_state == 'paused':
            snapshot = self._agent.observe_toolbox() if hasattr(self._agent, 'observe_toolbox') else {}
            self._last_tool_snapshot = snapshot
            self.last['tool_action'] = {'tool_id': 'none', 'command_id': 'paused'}
            return self._epoch.data() if self._epoch.data() else self._epoch.get_state()

        delay = 5.0
        if hasattr(self._agent, 'config'):
            cfg = self._agent.config()
            delay = float(cfg.get('personality', {}).get('recon_time', 5.0) or 5.0)
            if self.last.get('reflex_bias'):
                delay *= float(self.last['reflex_bias'].get('recon_time_multiplier', 1.0))
            delay = max(delay, 1.0)

        if not self._tool_actions:
            self.last['tool_action'] = dict(DEFAULT_TOOL_ACTION)
            snapshot = self._agent.observe_toolbox() if hasattr(self._agent, 'observe_toolbox') else {}
            self._last_tool_snapshot = snapshot
            self._epoch.observe(snapshot if snapshot else {'targets_visible': 0.0})

            logging.debug("[gym] no-tool sleep %.2fs", delay)
            self._epoch.track_sleep(delay)
            time.sleep(delay)

            # RL inner-loop epochs must stay lightweight; avoid full automata/plugin epoch hooks.
            self._epoch.next()
            return self._epoch.data() if self._epoch.data() else self._epoch.get_state()

        epoch_window, action_budget, inter_action_delay = self._epoch_runtime_budget(delay)
        logging.debug("[gym] epoch window=%.2fs action_budget=%d inter_action_delay=%.2fs", epoch_window, action_budget, inter_action_delay)
        epoch_started = time.time()
        epoch_deadline = epoch_started + epoch_window

        action_idx = int(policy if np.isscalar(policy) else np.asarray(policy).item())
        action_idx = self._sanitize_policy_action(action_idx)
        result, action = self._execute_action_index(action_idx)
        if action is None:
            self.last['tool_action'] = {'tool_id': 'none', 'command_id': 'paused'}
            return self._epoch.data() if self._epoch.data() else self._epoch.get_state()
        self.last['actions_executed'].append("%s.%s" % (action.tool_id, action.command_id))
        if 'ui' in result and isinstance(result['ui'], dict):
            self.last['ui_update'] = result['ui']

        while len(self.last['actions_executed']) < action_budget:
            if getattr(self._agent, '_ai_pause', False):
                break
            now = time.time()
            if now >= epoch_deadline:
                break

            follow_idx = self._choose_follow_up_action(self._last_tool_snapshot)
            if follow_idx is None:
                break

            remaining = max(epoch_deadline - now, 0.0)
            if inter_action_delay > 0 and remaining > 0:
                nap = min(inter_action_delay, remaining)

                time.sleep(nap)
                self._epoch.track_sleep(nap)

            result, action = self._execute_action_index(follow_idx)
            if action is None:
                break
            self.last['actions_executed'].append("%s.%s" % (action.tool_id, action.command_id))
            if 'ui' in result and isinstance(result['ui'], dict):
                self.last['ui_update'] = result['ui']

        if getattr(self._agent, '_ai_pause', False):
            return self._epoch.data() if self._epoch.data() else self._epoch.get_state()

        # Keep a short passive listen tail for handshake capture, but avoid long idle gaps.
        tail = min(max(epoch_deadline - time.time(), 0.0), 2.5)
        if tail > 0:
            logging.debug("[gym] step tail-sleep %.2fs", tail)

            self._epoch.track_sleep(tail)
            time.sleep(tail)

        # RL inner-loop epochs must stay lightweight; avoid full automata/plugin epoch hooks.
        logging.debug("[gym] post-exec: epoch.next begin")
        self._epoch.next()
        logging.debug("[gym] post-exec: epoch.next done")
        return self._epoch.data() if self._epoch.data() else self._epoch.get_state()

    def step(self, policy):
        logging.debug("[gym] step: starting step %d", self._epoch_num + 1)
        self._epoch_num += 1
        state = self._run_tool_step(policy)

        if self.reflex:
            reflex_input = dict(state)
            reflex_input['tool_metrics'] = dict(self._last_tool_snapshot.get('tool_metrics', {}))
            reflex_input['tool_costs'] = dict(self._last_tool_snapshot.get('tool_costs', {}))
            self.reflex.observe(reflex_input)

            self.last['reflex_bias'] = self.reflex.bias()
            self.last['physiology'] = {
                'tool_stress': self.reflex.stress_level,
                'tool_risk': self.reflex.risk,
            }
        else:
            self.last['reflex_bias'] = {}
            self.last['physiology'] = {}

        self.last['reward'] = state['reward']
        if self.reflex and self.last['physiology'].get('tool_risk', 0) > 0.9:
            logging.error("[ai] --- SYSTEM VIABILITY LOW: Reincarnation Likely ---")
            self.last['reward'] -= 5.0

        tool_action = self.last.get('tool_action') or dict(DEFAULT_TOOL_ACTION)
        if not isinstance(tool_action, dict):
            tool_action = dict(DEFAULT_TOOL_ACTION)
        self.last['tool_action'] = tool_action
        self.last['state'] = state
        self.last['state_v'] = featurizer.featurize(state, self._epoch_num)

        if hasattr(self._agent, '_view') and self._agent._view:
            self._agent._view.set('uptime', time.strftime('%H:%M:%S', time.gmtime(pwnagotchi.uptime())))

        self._agent.on_ai_step()

        if hasattr(self._agent, '_view') and self._agent._view:
            ui_update = self.last.get('ui_update')
            if ui_update:
                if 'face' in ui_update:
                    self._agent._view.set('face', ui_update['face'])
                if 'status' in ui_update:
                    self._agent._view.set('status', str(ui_update['status']))
                self._agent._view.update(force=True)

        logging.debug(
            "[gym] step complete: step=%d action=%s.%s actions=%d reward=%.6f",
            self._epoch_num,
            tool_action.get('tool_id', 'none'),
            tool_action.get('command_id', 'noop'),
            len(self.last.get('actions_executed') or []),
            float(self.last.get('reward', 0.0)),
        )

        terminated = not self._agent.is_training()
        truncated = False
        if PRIMAL:
            return self.last['state_v'], self.last['reward'], terminated, truncated, {}

        done = terminated
        return self.last['state_v'], self.last['reward'], done, {}

    def reset(self, seed=None, options=None):
        logging.debug("[gym] reset called")
        if PRIMAL:
            super().reset(seed=seed)

        self._epoch_num = 0
        state = self._epoch.data() if self._epoch.data() else self._epoch.get_state()
        self.last['state'] = state
        self.last['state_v'] = featurizer.featurize(state, 1)

        if PRIMAL:
            return self.last['state_v'], {}
        return self.last['state_v']

    def render(self, mode='human', close=False, force=False):
        if self._last_render == self._epoch_num:
            return

        if not self._agent.is_training() and not force:
            return

        self._last_render = self._epoch_num
        logging.info("[ai] --- training epoch %d/%d ---", self._epoch_num, self._agent.training_epochs())
        logging.info("[ai] REWARD: %f", self.last['reward'])

        if self.last.get('tool_action'):
            logging.info("[ai] ACTION: %s.%s", self.last['tool_action']['tool_id'], self.last['tool_action']['command_id'])
        if self.last.get('actions_executed'):
            logging.info("[ai] ACTIONS THIS EPOCH: %s", ', '.join(self.last['actions_executed']))

        if self.last.get('physiology'):
            logging.info(
                "[ai] PHYSIO: stress=%.2f risk=%.2f",
                self.last['physiology'].get('tool_stress', 0.0),
                self.last['physiology'].get('tool_risk', 0.0),
            )

        if self.last['state']:
            logging.info("[ai] observation:")
            for name in [
                'duration', 'interaction_rate', 'success_ratio', 'failure_ratio', 'error_ratio',
                'targets_visible', 'no_targets_for', 'context_switches',
                'active_for', 'inactive_for', 'sad_for', 'bored_for',
                'cpu_load', 'mem_usage', 'temperature',
            ]:
                if name in self.last['state']:
                    logging.info("    %s: %s", name, self.last['state'][name])
