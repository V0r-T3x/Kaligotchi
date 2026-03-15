import time
import threading
from typing import Any, Dict, Optional

import pwnagotchi

from pwnagotchi.kali.ai.reward import RewardFunction


class Epoch:
    """Tool-agnostic epoch state and reward accumulator."""

    def __init__(self, config: Dict[str, Any], reward_model=None):
        self.config = config
        self.reward_model = reward_model if reward_model is not None else RewardFunction()

        self.epoch_index = 0
        self.epoch = 0
        self.epoch_started = time.time()
        self.epoch_duration = 0.0

        self.num_interactions = 0
        self.num_success = 0
        self.num_failures = 0
        self.num_errors = 0

        self.targets_visible = 0
        self.no_targets_for = 0

        self.context_id = None
        self.context_switches = 0

        self.cpu_load = 0.0
        self.mem_usage = 0.0
        self.temperature = 0.0

        self.any_activity = False
        self.active_for = 0
        self.inactive_for = 0
        self.sad_for = 0
        self.bored_for = 0
        self.slept_for = 0.0

        self.last_reward = 0.0
        self.avg_reward = 0.0
        self.max_reward = -1e20
        self.min_reward = 1e20

        self._epoch_data = {}
        self._epoch_data_ready = threading.Event()
        # observe_snapshot() may call track_interaction() while holding this lock.
        # Use RLock to avoid self-deadlock on nested acquisition in the same thread.
        self._lock = threading.RLock()

    def wait_for_epoch_data(self, with_observation=True, timeout=None):
        self._epoch_data_ready.wait(timeout)
        self._epoch_data_ready.clear()
        return self._epoch_data

    def data(self):
        return self._epoch_data

    def observe(self, targets_visible: Any, context_id: Optional[Any] = None):
        """
        Accept either an int (targets visible) or a snapshot dict from tool manager.
        """
        if isinstance(targets_visible, dict):
            self.observe_snapshot(targets_visible)
            return

        with self._lock:
            self.targets_visible = int(targets_visible or 0)
            if self.targets_visible == 0:
                self.no_targets_for += 1
            else:
                self.no_targets_for = 0

            if context_id is not None:
                if self.context_id is not None and context_id != self.context_id:
                    self.context_switches += 1
                self.context_id = context_id

    def observe_snapshot(self, snapshot: Dict[str, Any]):
        with self._lock:
            self.targets_visible = int(snapshot.get('targets_visible', self.targets_visible) or 0)
            if self.targets_visible == 0:
                self.no_targets_for += 1
            else:
                self.no_targets_for = 0

            self.cpu_load = float(snapshot.get('cpu_load', self.cpu_load) or 0.0)
            self.mem_usage = float(snapshot.get('mem_usage', self.mem_usage) or 0.0)
            self.temperature = float(snapshot.get('temperature', self.temperature) or 0.0)

            if float(snapshot.get('context_switch', 0.0)) > 0:
                self.context_switches += 1

            success_count = int(round(float(snapshot.get('interaction_success', 0.0))))
            error_count = int(round(float(snapshot.get('interaction_error', 0.0))))
            if success_count > 0:
                self.track_interaction(success=True, error=False, count=success_count)
            if error_count > 0:
                self.track_interaction(success=False, error=True, count=error_count)

    def ingest_execution(self, execution: Dict[str, Any]):
        with self._lock:
            signals = execution.get('signals', {}) if isinstance(execution, dict) else {}
            success = bool(signals.get('interaction_success', 0) or execution.get('ok', False))
            has_error = bool(execution.get('error') or signals.get('execution_timeout', 0) or signals.get('resource_unavailable', 0))

        if success or has_error:
            self.track_interaction(success=success, error=has_error)

    def track_interaction(self, success: bool = False, error: bool = False, count: int = 1):
        with self._lock:
            count = int(count)
            if count <= 0:
                return

            self.num_interactions += count
            self.any_activity = True

            if success:
                self.num_success += count
            else:
                self.num_failures += count

            if error:
                self.num_errors += count

    def track_sleep(self, seconds: float):
        with self._lock:
            self.slept_for += float(seconds)

    def track(self, sleep=False, miss=False, success=False, error=False, inc=1, **_):
        if sleep:
            self.track_sleep(float(inc))
        elif miss:
            self.track_interaction(success=False, error=True, count=inc)
        else:
            self.track_interaction(success=bool(success), error=bool(error), count=inc)

    def update_system_metrics(self, cpu_load: float, mem_usage: float, temperature: float):
        self.cpu_load = float(cpu_load)
        self.mem_usage = float(mem_usage)
        self.temperature = float(temperature)

    def _update_activity_state(self):
        if not self.any_activity:
            self.inactive_for += 1
            self.active_for = 0
        else:
            self.active_for += 1
            self.inactive_for = 0

    def _update_emotional_state(self):
        boredom_threshold = int(self.config.get('boredom_threshold', self.config.get('personality', {}).get('bored_num_epochs', 5)))
        sadness_threshold = int(self.config.get('sadness_threshold', self.config.get('personality', {}).get('sad_num_epochs', 5)))

        if self.num_interactions == 0:
            self.bored_for += 1
        else:
            self.bored_for = 0

        if self.num_success == 0 and self.num_interactions > 0:
            self.sad_for += 1
        else:
            self.sad_for = 0

        if self.bored_for < boredom_threshold:
            self.bored_for = 0
        if self.sad_for < sadness_threshold:
            self.sad_for = 0

    def get_state(self) -> Dict[str, float]:
        duration = max(self.epoch_duration, 1e-6)

        success_ratio = self.num_success / self.num_interactions if self.num_interactions > 0 else 0.0
        failure_ratio = self.num_failures / self.num_interactions if self.num_interactions > 0 else 0.0
        error_ratio = self.num_errors / self.num_interactions if self.num_interactions > 0 else 0.0
        interaction_rate = self.num_interactions / duration

        return {
            'duration': duration,
            'interaction_rate': interaction_rate,
            'success_ratio': success_ratio,
            'failure_ratio': failure_ratio,
            'error_ratio': error_ratio,
            'targets_visible': float(self.targets_visible),
            'no_targets_for': float(self.no_targets_for),
            'context_switches': float(self.context_switches),
            'cpu_load': self.cpu_load,
            'mem_usage': self.mem_usage,
            'temperature': self.temperature,
            'active_for': float(self.active_for),
            'inactive_for': float(self.inactive_for),
            'sad_for': float(self.sad_for),
            'bored_for': float(self.bored_for),
            'slept_for': float(self.slept_for),
            'num_interactions': float(self.num_interactions),
            'num_success': float(self.num_success),
            'num_failures': float(self.num_failures),
            'num_errors': float(self.num_errors),
        }

    def finalize(self) -> float:
        print_marker = False
        with self._lock:
            try:
                import logging
                logging.warning("[kali.epoch] finalize begin")
                print_marker = True
            except Exception:
                pass
            self.epoch_duration = time.time() - self.epoch_started

            # Keep epoch finalization non-blocking for RL loop.
            # System metrics are expected to be provided by tool snapshots.
            if print_marker:
                logging.warning("[kali.epoch] finalize metrics skipped (snapshot-driven)")

            self._update_activity_state()
            self._update_emotional_state()

            if print_marker:
                logging.warning("[kali.epoch] finalize reward begin")
            state = self.get_state()
            self.last_reward = float(self.reward_model(self.epoch_index, state))
            self.avg_reward = ((self.avg_reward * self.epoch_index) + self.last_reward) / float(self.epoch_index + 1)
            self.max_reward = max(self.max_reward, self.last_reward)
            self.min_reward = min(self.min_reward, self.last_reward)

            state['reward'] = self.last_reward
            state['avg_reward'] = self.avg_reward
            state['max_reward'] = self.max_reward
            state['min_reward'] = self.min_reward

            self._epoch_data = state
            self._epoch_data_ready.set()

            self._reset_for_next_epoch()
            if print_marker:
                logging.warning("[kali.epoch] finalize done")
            return self.last_reward

    def next(self) -> float:
        return self.finalize()

    def _reset_for_next_epoch(self):
        self.epoch_index += 1
        self.epoch = self.epoch_index
        self.epoch_started = time.time()

        self.num_interactions = 0
        self.num_success = 0
        self.num_failures = 0
        self.num_errors = 0
        self.context_switches = 0
        self.any_activity = False
        self.slept_for = 0.0

    def __repr__(self):
        return (
            f"<Epoch #{self.epoch_index} | "
            f"interactions={self.num_interactions} "
            f"success={self.num_success} "
            f"errors={self.num_errors} "
            f"reward={self.last_reward:.3f}>"
        )
