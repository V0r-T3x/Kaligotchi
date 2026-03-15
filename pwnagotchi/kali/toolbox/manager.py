import importlib.util
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pwnagotchi
import pwnagotchi.kali.behavior_mapper as behavior_mapper
from pwnagotchi.kali.toolbox.voice_widget_manager import VoiceWidgetManager

try:
    import yaml
except Exception:
    yaml = None


@dataclass(frozen=True)
class ToolAction:
    tool_id: str
    command_id: str


DEFAULT_PERCEPTION_MEMORY = 10
MIN_PERCEPTION_MEMORY = 1
MAX_PERCEPTION_MEMORY = 64


class ManifestToolManager:
    """Loads tools from manifests and mediates abstract command execution."""

    def __init__(self, toolbox_root: str):
        self.toolbox_root = toolbox_root
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.adapters: Dict[str, Any] = {}

        self._active_tool_id: Optional[str] = None
        self._last_tool_id: Optional[str] = None
        self._last_metrics: Dict[str, float] = {}
        self._last_signals: Dict[str, Any] = {}
        self._last_costs: Dict[str, float] = {}
        self._perception_memory = DEFAULT_PERCEPTION_MEMORY
        self._observation_lock = threading.Lock()
        self._observation_history = deque(maxlen=self._perception_memory)
        self._command_cooldowns: Dict[str, float] = {}
        self._tool_failure_streaks: Dict[str, int] = {}
        self._tool_restart_ts: Dict[str, float] = {}
        self._runtime_config: Dict[str, Any] = {}
        self._view = None
        self._agent = None
        self._last_behavior: Optional[str] = None
        self._last_selected_target: Optional[str] = None
        self._voice_widget_manager = VoiceWidgetManager(toolbox_root)

        self.reload()

    def reload(self):
        self.tools = {}
        self.adapters = {}

        if not os.path.isdir(self.toolbox_root):
            logging.warning("[kali.toolbox] toolbox root not found: %s", self.toolbox_root)
            return

        for name in sorted(os.listdir(self.toolbox_root)):
            tool_dir = os.path.join(self.toolbox_root, name)
            manifest_dir = os.path.join(tool_dir, 'manifest')
            if not os.path.isdir(tool_dir) or not os.path.isdir(manifest_dir):
                continue

            tool_manifest = self._load_yaml(os.path.join(manifest_dir, 'tool.yaml'))
            if not tool_manifest:
                continue

            tool_id = str(tool_manifest.get('id') or name)
            commands = self._load_yaml(os.path.join(manifest_dir, 'commands.yaml')) or {}
            signals = self._load_yaml(os.path.join(manifest_dir, 'signals.yaml')) or {}
            metrics = self._load_yaml(os.path.join(manifest_dir, 'metrics.yaml')) or {}
            cost = self._load_yaml(os.path.join(manifest_dir, 'cost.yaml')) or {}
            reward = self._load_yaml(os.path.join(manifest_dir, 'reward.yaml')) or {}

            self.tools[tool_id] = {
                'id': tool_id,
                'dir': tool_dir,
                'manifest_dir': manifest_dir,
                'tool': tool_manifest,
                'commands': commands.get('commands', {}),
                'signals': signals,
                'metrics': metrics.get('metrics', {}),
                'cost': cost,
                'reward': reward,
            }

            adapter = self._load_adapter(tool_id, tool_dir)
            if adapter is not None:
                self.adapters[tool_id] = adapter

        if self._active_tool_id and self._active_tool_id not in self.tools:
            self._active_tool_id = None
            self._reset_observation_cache()

        logging.info("[kali.toolbox] loaded %d tool(s): %s", len(self.tools), ', '.join(self.tools.keys()))

    def list_tools(self) -> List[str]:
        return list(self.tools.keys())

    def active_tool(self) -> Optional[str]:
        return self._active_tool_id

    def tool_runtime_mode(self, tool_id: Optional[str] = None) -> str:
        tid = tool_id or self._active_tool_id or self._last_tool_id
        if not tid:
            return 'stopped'

        adapter = self.adapters.get(tid)
        if adapter is None:
            return 'stopped'

        if bool(getattr(adapter, 'running', False)):
            return 'active'

        if bool(getattr(adapter, '_loaded', False)):
            return 'paused'

        return 'stopped'

    def bind_view(self, view: Any):
        self._view = view
        self._voice_widget_manager.bind_view(view)

    def bind_agent(self, agent: Any):
        self._agent = agent

    def emit(self, tool_id: str, result: Optional[Dict[str, Any]] = None):
        if not tool_id or not isinstance(result, dict):
            return
        spec = self.tools.get(tool_id, {})
        self._voice_widget_manager.apply(tool_id, spec, result, self._runtime_config)

    def get_tool_ai_paths(self, tool_id: str, brains_root: str = '/root/brains') -> Dict[str, str]:
        spec = self.tools.get(tool_id, {})
        tool_manifest = spec.get('tool', {}) if isinstance(spec, dict) else {}
        ai_iface = tool_manifest.get('ai_interface', {}) if isinstance(tool_manifest, dict) else {}

        brain_file = str(ai_iface.get('brain_file', 'brain.nn'))
        reflex_file = str(ai_iface.get('reflex_file', 'reflex.pkl'))

        tool_dir = os.path.join(brains_root, tool_id)
        return {
            'tool_dir': tool_dir,
            'brain': os.path.join(tool_dir, os.path.basename(brain_file)),
            'reflex': os.path.join(tool_dir, os.path.basename(reflex_file)),
        }

    def list_actions(self) -> List[ToolAction]:
        actions: List[ToolAction] = []
        for tool_id, spec in self.tools.items():
            commands = spec.get('commands', {})
            for command_id, command_spec in commands.items():
                # Commands can opt out from policy action-space while remaining callable.
                if isinstance(command_spec, dict) and command_spec.get('ai_exposed', True) is False:
                    continue
                actions.append(ToolAction(tool_id=tool_id, command_id=command_id))
        return actions

    def activate_tool(self, tool_id: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._apply_runtime_config(context)
        if not tool_id:
            return {'ok': False, 'error': 'invalid_tool'}
        if tool_id not in self.tools:
            return {'ok': False, 'error': 'unknown_tool'}

        if self._active_tool_id:
            if self._active_tool_id != tool_id:
                self.deactivate_tool(self._active_tool_id, context=context)
            else:
                adapter = self.adapters.get(tool_id)
                try:
                    self._shutdown_adapter(adapter, context=context)
                except Exception as exc:
                    logging.warning("[kali.toolbox] tool shutdown failed: %s", exc)

        adapter = self.adapters.get(tool_id)
        result = self._call_adapter_hook(adapter, 'on_load', context=context)
        if result.get('ok', True):
            self._active_tool_id = tool_id
            self._last_tool_id = tool_id
        self.emit(tool_id, result)
        return result

    def pause_tool(self, tool_id: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Performs a soft unload (pause) of a tool if supported."""
        if not tool_id:
            return {'ok': True}

        adapter = self.adapters.get(tool_id)
        if not adapter:
            return {'ok': True, 'error': 'adapter_not_found'}

        # If adapter supports on_pause, use it for a soft unload.
        if hasattr(adapter, 'on_pause'):
            logging.info("[kali.toolbox] pausing tool %s", tool_id)
            result = self._call_adapter_hook(adapter, 'on_pause', context=context)
        else:
            # Fallback to hard unload for older adapters.
            logging.warning("[kali.toolbox] adapter for %s has no on_pause, falling back to full deactivation", tool_id)
            result = self.deactivate_tool(tool_id, context=context)
        self.emit(tool_id, result)
        return result

    def resume_tool(self, tool_id: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Resumes a previously paused tool."""
        if not tool_id or tool_id not in self.tools:
            return {'ok': False, 'error': 'unknown_tool'}

        adapter = self.adapters.get(tool_id)
        if not adapter or not hasattr(adapter, 'on_resume'):
            return self.activate_tool(tool_id, context=context)

        logging.info("[kali.toolbox] resuming tool %s", tool_id)
        result = self._call_adapter_hook(adapter, 'on_resume', context=context)
        if result.get('ok', True):
            self._active_tool_id = tool_id
            self._last_tool_id = tool_id
        self.emit(tool_id, result)
        return result

    def deactivate_tool(self, tool_id: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._apply_runtime_config(context)
        tid = tool_id or self._active_tool_id
        if not tid:
            self._active_tool_id = None
            self._reset_observation_cache()
            return {'ok': True}

        adapter = self.adapters.get(tid)
        result = self._shutdown_adapter(adapter, context=context)
        self.emit(tid, result)

        if tid == self._active_tool_id:
            self._active_tool_id = None

        self._reset_observation_cache()
        return result

    def restart_tool(self, tool_id: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._apply_runtime_config(context)
        tid = tool_id or self._active_tool_id
        if not tid:
            return {'ok': False, 'error': 'invalid_tool'}
        if tid not in self.tools:
            return {'ok': False, 'error': 'unknown_tool'}

        adapter = self.adapters.get(tid)
        if adapter is not None and hasattr(adapter, 'on_restart'):
            result = self._call_adapter_hook(adapter, 'on_restart', context=context)
            if result.get('ok', True):
                self._active_tool_id = tid
                self._last_tool_id = tid
                self.reset_environment_state(tool_id=tid, reason='tool_restart')
            return result

        down = self.deactivate_tool(tid, context=context)
        if not down.get('ok', False):
            return down
        return self.activate_tool(tid, context=context)

    def execute(self, action: ToolAction, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        context = context or {}
        self._apply_runtime_config(context)
        self._last_selected_target = str(context.get('target_mac') or '').strip().lower() or None

        if self._agent is not None and getattr(self._agent, 'state', 'running') != 'running':
            logging.debug("[kali.toolbox] agent idle, ignoring action")
            return {'ok': True, 'skipped': True, 'error': 'agent_idle', 'signals': {}, 'metrics': {}, 'costs': {}}

        if self._active_tool_id and action.tool_id != self._active_tool_id:
            return {'ok': False, 'error': 'tool_not_active', 'metrics': {}, 'signals': {}, 'costs': {}}

        spec = self.tools.get(action.tool_id)
        if not spec:
            return {'ok': False, 'error': 'unknown_tool', 'metrics': {}, 'signals': {}, 'costs': {}}

        command_spec = spec['commands'].get(action.command_id)
        if not command_spec:
            return {'ok': False, 'error': 'unknown_command', 'metrics': {}, 'signals': {}, 'costs': {}}

        cooldown_block = self._check_command_cooldown(action, command_spec)
        if cooldown_block is not None:
            self._last_tool_id = action.tool_id
            self._last_metrics = cooldown_block.get('metrics', {})
            self._last_signals = cooldown_block.get('signals', {})
            self._last_costs = cooldown_block.get('costs', {})
            self._push_observation_snapshot(action.tool_id, self._last_metrics, self._last_signals, self._last_costs)
            return cooldown_block

        started = time.time()
        adapter = self.adapters.get(action.tool_id)
        if adapter is None:
            logging.warning("[kali.toolbox] no active tool, ignoring action: %s", action.command_id)
            return {'ok': False, 'error': 'no_tool', 'signals': {}, 'metrics': {}, 'costs': {}}
        if hasattr(adapter, 'running') and not bool(getattr(adapter, 'running')):
            logging.warning("[kali.toolbox] tool not running, ignoring action: %s", action.command_id)
            return {'ok': False, 'error': 'tool_stopped', 'signals': {}, 'metrics': {}, 'costs': {}}

        result: Dict[str, Any] = {'ok': True, 'signals': {}, 'metrics': {}, 'costs': {}}

        if hasattr(adapter, 'execute'):
            try:
                logging.warning("[kali.toolbox] executing adapter %s command %s", action.tool_id, action.command_id)
                raw = adapter.execute(action.command_id, context, command_spec)
                if isinstance(raw, dict):
                    result.update(raw)
                else:
                    logging.warning("[kali.toolbox] adapter %s command %s returned non-dict: %s", action.tool_id, action.command_id, type(raw))
            except Exception as exc:
                logging.exception("[kali.toolbox] adapter execution failed (%s:%s): %s", action.tool_id, action.command_id, exc)
                result = {'ok': False, 'error': 'adapter_exception', 'signals': {'execution_timeout': 1.0}, 'metrics': {}, 'costs': {}}

        latency = max(time.time() - started, 0.0)
        raw_metrics = result.get('metrics', {}) if isinstance(result.get('metrics', {}), dict) else {}
        normalized = self._normalize_metrics(action.tool_id, result.get('signals', {}), latency, raw_metrics=raw_metrics)
        system_metrics = self._sample_system_metrics()
        merged_metrics = {**normalized, **raw_metrics, **system_metrics}
        failure_ratio = 1.0 if not bool(result.get('ok', False)) else 0.0
        merged_metrics['stress'] = max(
            0.0,
            min(
                (failure_ratio * 0.3)
                + (0.2 if float(merged_metrics.get('targets_visible', 0.0) or 0.0) <= 0.0 else 0.0)
                + (float(merged_metrics.get('cpu_load', 0.0) or 0.0) * 0.2)
                + (float(merged_metrics.get('temperature', 0.0) or 0.0) / 120.0),
                1.0,
            ),
        )
        costs = {**self._compute_costs(action.tool_id, merged_metrics, result.get('signals', {})), **result.get('costs', {})}

        result['metrics'] = merged_metrics
        result['costs'] = costs
        self._record_command_cooldown(action, command_spec)
        self._maybe_restart_tool(action.tool_id, spec, context, result)

        logging.info(
            "[kali.toolbox] action %s.%s ok=%s error=%s latency=%.3fs",
            action.tool_id,
            action.command_id,
            bool(result.get('ok', False)),
            result.get('error'),
            latency,
        )

        signals = result.get('signals', {}) if isinstance(result.get('signals', {}), dict) else {}
        reset_requested = bool(
            signals.get('environment_reset', 0.0)
            or signals.get('target_refresh_required', 0.0)
            or merged_metrics.get('environment_reset', 0.0)
            or merged_metrics.get('environment_unknown', 0.0)
        )
        if reset_requested:
            reset_reason = str(result.get('error') or merged_metrics.get('reset_reason') or 'environment_reset')
            reset_state = self.reset_environment_state(
                tool_id=action.tool_id,
                reason=reset_reason,
                system_metrics=system_metrics,
            )
            merged_metrics = {**merged_metrics, **reset_state.get('metrics', {})}
            costs = {**costs, **reset_state.get('costs', {})}
            signals.update(reset_state.get('signals', {}))
            result['metrics'] = merged_metrics
            result['costs'] = costs
            result['signals'] = signals

        self._last_tool_id = action.tool_id
        self._last_metrics = merged_metrics
        self._last_signals = signals
        self._last_costs = costs
        self._push_observation_snapshot(action.tool_id, merged_metrics, self._last_signals, costs)
        self._update_ui_behavior(action, command_spec, result)
        self._voice_widget_manager.apply(
            tool_id=action.tool_id,
            tool_spec=spec,
            result=result,
            runtime_config=self._runtime_config,
        )
        return result

    def observe(self) -> Dict[str, Any]:
        """Return generic snapshot for epoch + reflex (manifest-derived only)."""
        with self._observation_lock:
            history = list(self._observation_history)

        if len(history) <= 1:
            return self._build_observation_from_latest()

        tool_metrics = self._aggregate_history_mapping(history, 'metrics')
        tool_costs = self._aggregate_history_mapping(history, 'costs')
        interaction_success = self._aggregate_history_signals(history, ('interaction_success',), agg='mean')
        interaction_error = self._aggregate_history_signals(history, ('execution_timeout', 'resource_unavailable'), agg='sum')
        context_switch = self._aggregate_history_signals(history, ('context_switch',), agg='sum')

        targets_visible = self._aggregate_targets_visible(history)
        target_values = self._history_metric_values(history, 'targets_visible', fallback_blindness=True)
        if target_values:
            tool_metrics['stable_targets'] = float(sum(1 for value in target_values if value > 0.0))
            tool_metrics['environment_activity'] = self._aggregate_values(target_values, 'mean')
            tool_metrics['environment_stability'] = self._stddev(target_values)

        return {
            'tool_id': history[-1].get('tool_id') or self._last_tool_id,
            'targets_visible': targets_visible,
            'cpu_load': self._aggregate_history_metric(history, 'cpu_load', agg='mean'),
            'mem_usage': self._aggregate_history_metric(history, 'mem_usage', agg='mean'),
            'temperature': self._aggregate_history_metric(history, 'temperature', agg='mean'),
            'interaction_success': interaction_success,
            'interaction_error': interaction_error,
            'context_switch': context_switch,
            'environment_reset': float(history[-1].get('signals', {}).get('environment_reset', 0.0)),
            'target_refresh_required': float(history[-1].get('signals', {}).get('target_refresh_required', 0.0)),
            'tool_metrics': tool_metrics,
            'tool_costs': tool_costs,
        }

    def _call_adapter_hook(self, adapter: Any, hook: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if adapter is None or not hasattr(adapter, hook):
            return {'ok': True}

        fn = getattr(adapter, hook)
        if not callable(fn):
            return {'ok': False, 'error': f'invalid_{hook}_hook'}

        try:
            out = fn(context=context or {})
            if isinstance(out, dict):
                return {'ok': bool(out.get('ok', True)), **out}
            return {'ok': bool(out) if out is not None else True}
        except Exception as exc:
            logging.exception("[kali.toolbox] lifecycle hook %s failed: %s", hook, exc)
            return {'ok': False, 'error': 'lifecycle_exception', 'exception': str(exc)}

    def _shutdown_adapter(self, adapter: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        unload = self._call_adapter_hook(adapter, 'on_unload', context=context)
        shutdown = self._call_adapter_hook(adapter, 'shutdown', context=context)
        if not unload.get('ok', True):
            return unload
        if isinstance(unload.get('events'), dict) and not isinstance(shutdown.get('events'), dict):
            shutdown['events'] = unload.get('events')
        return shutdown

    def _reset_observation_cache(self):
        self._last_tool_id = None
        self._last_metrics = {}
        self._last_signals = {}
        self._last_costs = {}
        self._last_selected_target = None
        self._tool_failure_streaks.clear()
        with self._observation_lock:
            self._observation_history.clear()

    def reset_environment_state(
        self,
        tool_id: Optional[str] = None,
        reason: str = 'environment_reset',
        system_metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        metrics = {
            'targets_visible': 0.0,
            'ap_count': 0.0,
            'client_count': 0.0,
            'environment_activity': 0.0,
            'environment_unknown': 1.0,
            'environment_reset': 1.0,
            'interaction_rate': 0.0,
            'blind_for_epochs': 1.0,
        }
        if isinstance(system_metrics, dict):
            metrics.update({
                'cpu_load': float(system_metrics.get('cpu_load', 0.0) or 0.0),
                'mem_usage': float(system_metrics.get('mem_usage', 0.0) or 0.0),
                'temperature': float(system_metrics.get('temperature', 0.0) or 0.0),
            })

        signals = {
            'environment_reset': 1.0,
            'target_refresh_required': 1.0,
            'interaction_success': 0.0,
            'interaction_failures': 1.0,
        }
        self._last_selected_target = None
        self._tool_failure_streaks.clear()
        self._last_tool_id = tool_id or self._last_tool_id
        self._last_metrics = metrics
        self._last_signals = signals
        self._last_costs = {}
        with self._observation_lock:
            self._observation_history.clear()
        logging.info("[kali.toolbox] environment state reset: tool=%s reason=%s", self._last_tool_id, reason)
        return {
            'tool_id': self._last_tool_id,
            'metrics': metrics,
            'signals': signals,
            'costs': {},
            'reason': reason,
        }

    def _apply_runtime_config(self, context: Optional[Dict[str, Any]] = None):
        config = context.get('config', {}) if isinstance(context, dict) else {}
        self._runtime_config = config if isinstance(config, dict) else {}
        kali_cfg = config.get('kali', {}) if isinstance(config, dict) else {}
        raw_window = kali_cfg.get('perception_memory', DEFAULT_PERCEPTION_MEMORY) if isinstance(kali_cfg, dict) else DEFAULT_PERCEPTION_MEMORY
        try:
            window = int(raw_window)
        except (TypeError, ValueError):
            window = DEFAULT_PERCEPTION_MEMORY

        window = max(MIN_PERCEPTION_MEMORY, min(window, MAX_PERCEPTION_MEMORY))
        if window == self._perception_memory:
            return

        with self._observation_lock:
            self._perception_memory = window
            self._observation_history = deque(self._observation_history, maxlen=window)

    def _push_observation_snapshot(
        self,
        tool_id: str,
        metrics: Dict[str, Any],
        signals: Dict[str, Any],
        costs: Dict[str, Any],
    ):
        snapshot = {
            'tool_id': tool_id,
            'metrics': dict(metrics or {}),
            'signals': dict(signals or {}),
            'costs': dict(costs or {}),
        }
        with self._observation_lock:
            self._observation_history.append(snapshot)

    def _build_observation_from_latest(self) -> Dict[str, Any]:
        blindness = float(self._last_metrics.get('blindness', 0.0))
        targets_visible = float(self._last_metrics.get('targets_visible', 0.0))
        if targets_visible <= 0.0 and blindness > 0.0:
            targets_visible = max(0.0, 1.0 - blindness)

        return {
            'tool_id': self._last_tool_id,
            'targets_visible': targets_visible,
            'cpu_load': float(self._last_metrics.get('cpu_load', 0.0)),
            'mem_usage': float(self._last_metrics.get('mem_usage', 0.0)),
            'temperature': float(self._last_metrics.get('temperature', 0.0)),
            'interaction_success': float(self._last_signals.get('interaction_success', 0.0)),
            'interaction_error': float(self._last_signals.get('execution_timeout', 0.0) or self._last_signals.get('resource_unavailable', 0.0)),
            'context_switch': float(self._last_signals.get('context_switch', 0.0)),
            'environment_reset': float(self._last_signals.get('environment_reset', 0.0)),
            'target_refresh_required': float(self._last_signals.get('target_refresh_required', 0.0)),
            'tool_metrics': dict(self._last_metrics),
            'tool_costs': dict(self._last_costs),
        }

    def _sample_system_metrics(self) -> Dict[str, float]:
        metrics = {
            'cpu_load': 0.0,
            'mem_usage': 0.0,
            'temperature': 0.0,
        }

        try:
            metrics['cpu_load'] = float(pwnagotchi.cpu_load('kali.toolbox'))
        except Exception:
            pass

        try:
            metrics['mem_usage'] = float(pwnagotchi.mem_usage())
        except Exception:
            pass

        try:
            metrics['temperature'] = float(pwnagotchi.temperature())
        except Exception:
            pass

        return metrics

    def _update_ui_behavior(self, action: ToolAction, command_spec: Dict[str, Any], result: Dict[str, Any]):
        observation = self.observe()
        behavior = behavior_mapper.action_to_behavior(
            action=action,
            command_spec=command_spec,
            result=result,
            observation=observation,
        )
        self._last_behavior = behavior

        try:
            if self._agent is not None and hasattr(self._agent, 'set_behavior'):
                self._agent.set_behavior(behavior)
            explicit_voice = isinstance(result.get('events', {}), dict) and bool(result.get('events', {}).get('voice_event'))
            if not explicit_voice and self._agent is not None and hasattr(self._agent, '_update_ui'):
                self._agent._update_ui(observation)
        except Exception:
            logging.debug("[kali.toolbox] failed to propagate behavior=%s", behavior, exc_info=True)

    def _check_command_cooldown(self, action: ToolAction, command_spec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        cooldown = float(command_spec.get('cooldown_seconds', 0.0) or 0.0)
        if cooldown <= 0.0:
            return None

        key = self._command_cooldown_key(action, command_spec)
        now = time.time()
        last_run = float(self._command_cooldowns.get(key, 0.0) or 0.0)
        remaining = cooldown - (now - last_run)
        if remaining <= 0.0:
            return None

        logging.info(
            "[kali.toolbox] action %s.%s suppressed by cooldown %.1fs remaining",
            action.tool_id,
            action.command_id,
            remaining,
        )
        return {
            'ok': False,
            'error': 'cooldown_active',
            'signals': {
                'cooldown_blocked': 1.0,
                'interaction_success': 0.0,
                'interaction_failures': 0.0,
            },
            'metrics': {
                'cooldown_remaining': float(remaining),
            },
            'costs': {},
        }

    def _record_command_cooldown(self, action: ToolAction, command_spec: Dict[str, Any]):
        cooldown = float(command_spec.get('cooldown_seconds', 0.0) or 0.0)
        if cooldown <= 0.0:
            return
        self._command_cooldowns[self._command_cooldown_key(action, command_spec)] = time.time()

    @staticmethod
    def _command_cooldown_key(action: ToolAction, command_spec: Dict[str, Any]) -> str:
        group = str(command_spec.get('cooldown_group', action.command_id) or action.command_id)
        return "%s:%s" % (action.tool_id, group)

    def _maybe_restart_tool(
        self,
        tool_id: str,
        spec: Dict[str, Any],
        context: Dict[str, Any],
        result: Dict[str, Any],
    ):
        signals = result.get('signals', {}) if isinstance(result.get('signals', {}), dict) else {}
        recovery_cfg = spec.get('tool', {}).get('recovery', {}) if isinstance(spec.get('tool', {}), dict) else {}
        if not isinstance(recovery_cfg, dict) or not bool(recovery_cfg.get('enabled', False)):
            return

        restart_signal = str(recovery_cfg.get('restart_signal', 'restart_recommended') or 'restart_recommended')
        restart_threshold = int(recovery_cfg.get('failure_threshold', 3) or 3)
        restart_cooldown = float(recovery_cfg.get('cooldown_seconds', 90.0) or 90.0)
        should_restart = bool(signals.get(restart_signal, 0.0))

        if result.get('ok', False):
            self._tool_failure_streaks[tool_id] = 0
            return

        failures = int(self._tool_failure_streaks.get(tool_id, 0)) + 1
        self._tool_failure_streaks[tool_id] = failures
        should_restart = should_restart or failures >= max(restart_threshold, 1)
        if not should_restart:
            return

        last_restart = float(self._tool_restart_ts.get(tool_id, 0.0) or 0.0)
        if (time.time() - last_restart) < restart_cooldown:
            return

        recovery_context = dict(context or {})
        if 'config' not in recovery_context and hasattr(self, '_runtime_config'):
            recovery_context['config'] = self._runtime_config

        restart_result = self.restart_tool(tool_id, context=recovery_context)
        self._tool_restart_ts[tool_id] = time.time()
        signals['tool_restart_attempted'] = 1.0
        signals['tool_restart_success'] = 1.0 if restart_result.get('ok', False) else 0.0
        if restart_result.get('ok', False):
            self._tool_failure_streaks[tool_id] = 0
        else:
            signals['module_instability'] = max(float(signals.get('module_instability', 0.0)), 1.0)

    def _aggregate_targets_visible(self, history: List[Dict[str, Any]]) -> float:
        values = self._history_metric_values(history, 'targets_visible', fallback_blindness=True)
        return float(values[-1]) if values else 0.0

    def _aggregate_history_metric(self, history: List[Dict[str, Any]], key: str, agg: str = 'mean') -> float:
        values = self._history_metric_values(history, key)
        return self._aggregate_values(values, agg)

    def _history_metric_values(
        self,
        history: List[Dict[str, Any]],
        key: str,
        fallback_blindness: bool = False,
    ) -> List[float]:
        values: List[float] = []
        for entry in history:
            metrics = entry.get('metrics', {})
            if key in metrics:
                values.append(float(metrics.get(key, 0.0)))
                continue
            if fallback_blindness and key == 'targets_visible' and 'blindness' in metrics:
                blindness = float(metrics.get('blindness', 0.0))
                values.append(max(0.0, 1.0 - blindness) if blindness > 0.0 else 0.0)
        return values

    def _aggregate_history_signals(self, history: List[Dict[str, Any]], keys: tuple, agg: str) -> float:
        values: List[float] = []
        for entry in history:
            signals = entry.get('signals', {})
            entry_value = None
            for key in keys:
                if key in signals:
                    entry_value = float(signals.get(key, 0.0))
                    if entry_value:
                        break
            if entry_value is not None:
                values.append(entry_value)
        return self._aggregate_values(values, agg)

    def _aggregate_history_mapping(self, history: List[Dict[str, Any]], field: str) -> Dict[str, float]:
        values_by_key: Dict[str, List[float]] = {}
        for entry in history:
            mapping = entry.get(field, {})
            if not isinstance(mapping, dict):
                continue
            for key, value in mapping.items():
                try:
                    values_by_key.setdefault(key, []).append(float(value))
                except (TypeError, ValueError):
                    continue

        aggregated: Dict[str, float] = {}
        for key, values in values_by_key.items():
            if not values:
                continue
            if key == 'targets_visible':
                aggregated[key] = float(values[-1])
            elif key in ('cpu_load', 'mem_usage', 'temperature', 'command_latency', 'friction', 'stress', 'risk'):
                aggregated[key] = self._aggregate_values(values, 'mean')
            else:
                aggregated[key] = self._aggregate_values(values, 'mean')
        return aggregated

    @staticmethod
    def _aggregate_values(values: List[float], agg: str) -> float:
        if not values:
            return 0.0
        if agg == 'sum':
            return float(sum(values))
        if agg == 'max':
            return float(max(values))
        return float(sum(values) / len(values))

    @staticmethod
    def _stddev(values: List[float]) -> float:
        if len(values) <= 1:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        return float(variance ** 0.5)

    def _load_yaml(self, path: str) -> Dict[str, Any]:
        if not os.path.isfile(path):
            return {}
        if yaml is None:
            logging.error("[kali.toolbox] PyYAML is required to parse %s", path)
            return {}

        try:
            with open(path, 'rt', encoding='utf-8') as fp:
                data = yaml.safe_load(fp) or {}
                if not isinstance(data, dict):
                    return {}
                return data
        except Exception as exc:
            logging.error("[kali.toolbox] failed reading %s: %s", path, exc)
            return {}

    def _load_adapter(self, tool_id: str, tool_dir: str):
        tool_py = os.path.join(tool_dir, 'tool.py')
        if not os.path.isfile(tool_py):
            return None

        try:
            module_name = f"kali_tool_{tool_id}"
            spec = importlib.util.spec_from_file_location(module_name, tool_py)
            if spec is None or spec.loader is None:
                return None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if hasattr(module, 'ToolAdapter'):
                return module.ToolAdapter(tool_dir=tool_dir)
            if hasattr(module, 'adapter'):
                return module.adapter
            logging.warning("[kali.toolbox] %s loaded but no ToolAdapter class or adapter instance found", tool_py)
            return None
        except Exception as exc:
            logging.error("[kali.toolbox] unable to load adapter for %s: %s", tool_id, exc)
            return None

    def _normalize_metrics(
        self,
        tool_id: str,
        signals: Dict[str, Any],
        latency: float,
        raw_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        spec = self.tools.get(tool_id, {})
        metrics_spec = spec.get('metrics', {})
        raw_metrics = raw_metrics or {}

        out: Dict[str, float] = {'command_latency': float(latency)}

        for metric_name, metric_rule in metrics_spec.items():
            source = str(metric_rule.get('source', '')).strip()
            transform = metric_rule.get('transform', {})
            if source:
                raw_value = float(raw_metrics.get(source, signals.get(source, 0.0)))
            else:
                raw_value = 0.0
            out[metric_name] = self._apply_transform(raw_value, transform)

        return out

    def _compute_costs(self, tool_id: str, metrics: Dict[str, float], signals: Dict[str, Any]) -> Dict[str, float]:
        spec = self.tools.get(tool_id, {})
        cost_spec = spec.get('cost', {})
        costs: Dict[str, float] = {}

        for dim in ('stress', 'risk'):
            dim_spec = cost_spec.get(dim, {}) if isinstance(cost_spec, dict) else {}
            sources = dim_spec.get('sources', {}) if isinstance(dim_spec, dict) else {}
            values: List[float] = []
            for source_name, source_spec in sources.items():
                source_spec = source_spec or {}
                maximum = float(source_spec.get('max', 1.0) or 1.0)
                weight = float(source_spec.get('weight', 1.0) or 1.0)
                raw = float(metrics.get(source_name, signals.get(source_name, 0.0)))
                values.append(max(0.0, min(raw / maximum, 1.0)) * weight)

            if not values:
                score = 0.0
            else:
                agg = str(dim_spec.get('aggregation', 'max'))
                if agg == 'sum':
                    score = sum(values)
                elif agg == 'mean':
                    score = sum(values) / len(values)
                else:
                    score = max(values)

            clamp = float(dim_spec.get('clamp', 1.0) or 1.0)
            costs[dim] = max(0.0, min(score, clamp))

        friction = float(metrics.get('friction', 0.0))
        costs['friction'] = max(0.0, min(friction, 1.0))
        return costs

    @staticmethod
    def _apply_transform(value: float, transform: Dict[str, Any]) -> float:
        t = str(transform.get('type', 'linear'))

        if t == 'linear':
            maximum = float(transform.get('max', 1.0) or 1.0)
            return max(0.0, min(value / maximum, 1.0))

        if t == 'threshold':
            safe = float(transform.get('safe', 0.0))
            rng = float(transform.get('range', 1.0) or 1.0)
            return max(0.0, min((value - safe) / rng, 1.0))

        if t == 'clamp':
            maximum = float(transform.get('max', 1.0) or 1.0)
            return max(0.0, min(value, maximum))

        return max(0.0, min(value, 1.0))
