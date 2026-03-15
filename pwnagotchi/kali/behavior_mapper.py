from typing import Any, Dict, Optional

import pwnagotchi.ui.faces as faces


ACTION_BEHAVIORS = {
    "bettercap.explore_environment": "searching",
    "bettercap.set_channel": "searching",
    "bettercap.focus_context": "searching",
    "bettercap.interact_with_target": "interacting",
    "bettercap.force_state_change": "recon",
    "bettercap.sync_events": "thinking",
    "bettercap.clear_state": "recovering",
    "bettercap.restart_recon": "recovering",
    "bettercap.stop_recon": "recovering",
}


BEHAVIOR_FACES = {
    "searching": faces.LOOK_R,
    "interacting": faces.EXCITED,
    "recon": faces.LOOK_L,
    "thinking": faces.INTENSE,
    "idle": faces.AWAKE,
    "recovering": faces.BROKEN,
}


BEHAVIOR_STATUS = {
    "searching": "Scanning channels...",
    "interacting": "Probing target...",
    "recon": "Forcing recon...",
    "thinking": "Analyzing environment...",
    "idle": "Standing by...",
    "recovering": "Recovering radio...",
}


def action_to_behavior(
    action: Any,
    command_spec: Optional[Dict[str, Any]] = None,
    result: Optional[Dict[str, Any]] = None,
    observation: Optional[Dict[str, Any]] = None,
) -> str:
    events = result.get('events', {}) if isinstance(result, dict) else {}
    voice_event = str(events.get('voice_event') or '').strip()
    if voice_event in ('tool_paused',):
        return 'paused'
    if voice_event in ('tool_unloaded',):
        return 'stopped'
    if voice_event in ('radio_recovering', 'resource_unavailable', 'timeout'):
        return 'recovering'

    tool_id = getattr(action, 'tool_id', None)
    command_id = getattr(action, 'command_id', None)
    action_id = "%s.%s" % (tool_id, command_id) if tool_id and command_id else ""
    if action_id in ACTION_BEHAVIORS:
        return ACTION_BEHAVIORS[action_id]

    signals = result.get('signals', {}) if isinstance(result, dict) else {}
    if float(signals.get('tool_restart_attempted', 0.0) or 0.0) > 0.0:
        return "recovering"

    command_spec = command_spec if isinstance(command_spec, dict) else {}
    capability = str(command_spec.get('capability', '') or '').lower()
    intent = str(command_spec.get('intent', '') or '').lower()

    if intent in ('recover', 'reset') or capability == 'maintenance':
        return "recovering" if intent == 'recover' else "thinking"
    if capability in ('exploration', 'optimization') or intent in ('discover', 'focus'):
        return "searching"
    if capability == 'interaction' or intent == 'engage':
        return "interacting"
    if capability == 'disruption' or intent in ('provoke', 'attack'):
        return "recon"
    if capability == 'control' or intent in ('configure', 'sync', 'analyze', 'pause'):
        return "thinking"

    obs = observation if isinstance(observation, dict) else {}
    if float(obs.get('targets_visible', 0.0) or 0.0) > 0.0:
        return "searching"
    return "idle"


def behavior_to_face(behavior: str, observation: Optional[Dict[str, Any]] = None) -> str:
    obs = observation if isinstance(observation, dict) else {}
    tool_costs = obs.get('tool_costs', {}) if isinstance(obs.get('tool_costs', {}), dict) else {}
    tool_metrics = obs.get('tool_metrics', {}) if isinstance(obs.get('tool_metrics', {}), dict) else {}
    stress = float(tool_costs.get('stress', 0.0) or 0.0)
    risk = float(tool_costs.get('risk', 0.0) or 0.0)
    instability = float(tool_metrics.get('module_instability', 0.0) or 0.0)
    interaction_failures = float(tool_metrics.get('interaction_failures', 0.0) or 0.0)
    restart_recommended = float(tool_metrics.get('restart_recommended', 0.0) or 0.0)
    targets_visible = float(obs.get('targets_visible', 0.0) or 0.0)
    environment_activity = float(tool_metrics.get('environment_activity', 0.0) or 0.0)

    if behavior == 'stopped':
        return faces.SLEEP
    if behavior == 'paused':
        return faces.SLEEP2
    if behavior == 'recovering' or instability > 0.7 or restart_recommended > 0.0:
        return BEHAVIOR_FACES['recovering']
    if stress > 0.8 and risk > 0.8:
        return faces.ANGRY
    if targets_visible > 0.0 and interaction_failures <= 0.0 and behavior in ('searching', 'interacting'):
        return faces.EXCITED
    if behavior == 'searching' and stress < 0.5 and risk < 0.5:
        return faces.LOOK_R
    if targets_visible <= 0.0 and environment_activity <= 0.0 and behavior == 'idle':
        return faces.BORED
    return BEHAVIOR_FACES.get(behavior, faces.AWAKE)


def build_status(behavior: str, observation: Optional[Dict[str, Any]] = None) -> str:
    obs = observation if isinstance(observation, dict) else {}
    targets_visible = int(float(obs.get('targets_visible', 0.0) or 0.0))
    tool_metrics = obs.get('tool_metrics', {}) if isinstance(obs.get('tool_metrics', {}), dict) else {}
    tool_costs = obs.get('tool_costs', {}) if isinstance(obs.get('tool_costs', {}), dict) else {}
    stable_targets = int(float(tool_metrics.get('stable_targets', 0.0) or 0.0))
    stress = float(tool_costs.get('stress', 0.0) or 0.0)
    risk = float(tool_costs.get('risk', 0.0) or 0.0)
    instability = float(tool_metrics.get('module_instability', 0.0) or 0.0)
    restart_recommended = float(tool_metrics.get('restart_recommended', 0.0) or 0.0)

    base = BEHAVIOR_STATUS.get(behavior, BEHAVIOR_STATUS['idle'])
    if behavior == 'paused':
        return "Tool paused, runtime preserved"
    if behavior == 'stopped':
        return "No active tool, runtime stopped"
    if instability > 0.7 or restart_recommended > 0.0:
        return "Radio recovering..."
    if stress > 0.8 and risk > 0.8:
        return "Pressure spike detected..."
    if behavior == 'searching' and targets_visible > 0:
        return "Tracking %d target%s..." % (targets_visible, '' if targets_visible == 1 else 's')
    if behavior == 'interacting' and targets_visible > 0:
        return "Probing %d live target%s..." % (targets_visible, '' if targets_visible == 1 else 's')
    if behavior == 'thinking' and stable_targets > 0:
        return "Analyzing %d stable target%s..." % (stable_targets, '' if stable_targets == 1 else 's')
    return base
