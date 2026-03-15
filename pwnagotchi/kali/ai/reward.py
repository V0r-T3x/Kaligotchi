import math

range = (-1.0, 1.0)
_eps = 1e-9


class RewardFunction(object):
    """Tool-agnostic default reward model."""

    @staticmethod
    def _get(state, key, default=0.0):
        return float(state.get(key, default))

    def __call__(self, epoch_n, state):
        epochs = float(epoch_n + 1)

        duration = max(self._get(state, 'duration', 1.0), _eps)
        interaction_rate = self._get(state, 'interaction_rate', self._get(state, 'num_interactions', 0.0) / duration)

        success_ratio = self._get(state, 'success_ratio', 0.0)
        error_ratio = self._get(state, 'error_ratio', 0.0)

        no_targets_for = self._get(state, 'no_targets_for', 0.0)
        inactive_for = self._get(state, 'inactive_for', 0.0)
        active_for = self._get(state, 'active_for', 0.0)
        sad_for = self._get(state, 'sad_for', 0.0)
        bored_for = self._get(state, 'bored_for', 0.0)
        context_switches = self._get(state, 'context_switches', 0.0)

        cpu_load = self._get(state, 'cpu_load', 0.0)
        mem_usage = self._get(state, 'mem_usage', 0.0)
        temperature = self._get(state, 'temperature', 0.0)
        temp_norm = min(temperature / 100.0, 1.0) if temperature > 1.0 else max(min(temperature, 1.0), 0.0)
        resource_pressure = min(max((cpu_load + mem_usage + temp_norm) / 3.0, 0.0), 1.0)

        active_norm = active_for / max(epochs, 1.0)
        inactive_norm = inactive_for / max(epochs, 1.0)
        no_target_norm = no_targets_for / max(epochs, 1.0)
        sad_norm = sad_for / max(epochs, 1.0)
        bored_norm = bored_for / max(epochs, 1.0)

        score = 0.0
        score += 0.50 * success_ratio
        score += 0.20 * math.tanh(interaction_rate)
        score += 0.10 * min(context_switches / 5.0, 1.0)
        score += 0.10 * min(active_norm, 1.0)

        score -= 0.30 * error_ratio
        score -= 0.20 * min(no_target_norm, 1.0)
        score -= 0.20 * min(inactive_norm, 1.0)
        score -= 0.10 * min(sad_norm, 1.0)
        score -= 0.05 * min(bored_norm, 1.0)
        score -= 0.05 * resource_pressure

        lo, hi = range
        return max(lo, min(hi, score))


class GenericToolReward:
    """Compatibility wrapper around RewardFunction."""

    def __init__(self, config):
        self.cfg = config
        self._model = RewardFunction()

    def compute(self, epoch_n, state):
        return self._model(epoch_n, state)
