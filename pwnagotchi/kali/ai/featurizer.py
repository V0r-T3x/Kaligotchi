import numpy as np

MAX_EPOCH_DURATION = 1024.0


def describe(extended=False):
    feature_count = 15
    return 0, (1, feature_count)


def featurize(state, step):
    epochs = float(step) + 1e-10

    duration = float(state.get('duration', 0.0))
    interaction_rate = float(state.get('interaction_rate', 0.0))
    success_ratio = float(state.get('success_ratio', 0.0))
    failure_ratio = float(state.get('failure_ratio', 0.0))
    error_ratio = float(state.get('error_ratio', 0.0))

    targets_visible = float(state.get('targets_visible', 0.0))
    no_targets_for = float(state.get('no_targets_for', 0.0))
    context_switches = float(state.get('context_switches', 0.0))

    cpu_load = float(state.get('cpu_load', 0.0))
    mem_usage = float(state.get('mem_usage', 0.0))
    temperature = float(state.get('temperature', 0.0))
    temp_norm = min(temperature / 100.0, 1.0) if temperature > 1.0 else max(min(temperature, 1.0), 0.0)

    active_for = float(state.get('active_for', 0.0))
    inactive_for = float(state.get('inactive_for', 0.0))
    sad_for = float(state.get('sad_for', 0.0))
    bored_for = float(state.get('bored_for', 0.0))

    return np.array([
        np.clip(duration / MAX_EPOCH_DURATION, 0.0, 1.0),
        np.clip(np.tanh(interaction_rate), -1.0, 1.0),
        np.clip(success_ratio, 0.0, 1.0),
        np.clip(failure_ratio, 0.0, 1.0),
        np.clip(error_ratio, 0.0, 1.0),
        np.clip(targets_visible / 50.0, 0.0, 1.0),
        np.clip(no_targets_for / epochs, 0.0, 1.0),
        np.clip(context_switches / epochs, 0.0, 1.0),
        np.clip(cpu_load, 0.0, 1.0),
        np.clip(mem_usage, 0.0, 1.0),
        np.clip(temp_norm, 0.0, 1.0),
        np.clip(active_for / epochs, 0.0, 1.0),
        np.clip(inactive_for / epochs, 0.0, 1.0),
        np.clip(sad_for / epochs, 0.0, 1.0),
        np.clip(bored_for / epochs, 0.0, 1.0),
    ], dtype=np.float32)
