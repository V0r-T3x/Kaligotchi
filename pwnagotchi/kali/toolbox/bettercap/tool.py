import logging
import os
import re
import string
import subprocess
import time
from collections import deque
from typing import Any, Dict, List, Optional, Set

try:
    import yaml
except Exception:
    yaml = None

from pwnagotchi.kali.client import Client


class ToolAdapter:
    """Bettercap adapter that executes manifest commands via Client.run()."""

    def __init__(self, tool_dir: str):
        self.tool_dir = tool_dir
        self.manifest_dir = os.path.join(tool_dir, 'manifest')
        self.tool_manifest = self._load_yaml('tool.yaml')
        self.client_manifest = self._load_yaml('client.yaml')

        client_cfg = (self.client_manifest.get('client') or {})
        self.defaults = (self.client_manifest.get('defaults') or {})
        self._runtime_defaults: Dict[str, Any] = {}
        self._loaded = False
        self.running = False
        self._mac_to_ssid: Dict[str, str] = {}

        self.client = Client(
            hostname=str(client_cfg.get('hostname', 'localhost')),
            scheme=str(client_cfg.get('scheme', 'http')),
            port=int(client_cfg.get('port', 8081)),
            username=str(client_cfg.get('username', 'user')),
            password=str(client_cfg.get('password', 'pass')),
            request_timeout=int(client_cfg.get('request_timeout', 10)),
        )
        self._known_handshakes: Set[str] = set()
        self._seen_journal_handshakes: Set[str] = set()
        self._last_session_trophy_name = 'none'
        self._last_session_trophy_bssid = ''
        self._target_cursor = 0
        self._last_journal_probe_ts = max(time.time() - 2.0, 0.0)
        self._api_failure_streak = 0
        self._deauth_window = deque(maxlen=20)
        self._last_restart_signal_ts = 0.0
        self._last_connection_errors = 0
        self._last_wifi_snapshot: Dict[str, Any] = {
            'aps': {},
            'clients': {},
            'channel_activity': {},
        }
        self._last_discovery_ts = time.time()
        self._last_recon_refresh_ts = 0.0
        self._channel_history = deque(maxlen=24)
        self._active_channel: Optional[int] = None

    def _reset_session_pwnd(self, reason: str) -> None:
        self._known_handshakes.clear()
        self._seen_journal_handshakes.clear()
        self._last_session_trophy_name = ''
        self._last_session_trophy_bssid = ''
        self._active_channel = None
        logging.info("[bettercap.tool] session PWND reset on %s", reason)

    @staticmethod
    def _capture_basename(path: str) -> str:
        return os.path.splitext(os.path.basename(str(path or '')))[0].strip().lower()

    def _list_unique_capture_files(self, handshakes_path: str) -> List[str]:
        if not handshakes_path or not os.path.isdir(handshakes_path):
            return []

        allowed_exts = {'.pcap', '.pcapng', '.cap'}
        latest_by_base: Dict[str, str] = {}
        latest_mtime: Dict[str, float] = {}

        try:
            for entry in os.scandir(handshakes_path):
                if not entry.is_file():
                    continue
                ext = os.path.splitext(entry.name)[1].lower()
                if ext not in allowed_exts:
                    continue

                base = self._capture_basename(entry.name)
                if not base:
                    continue

                try:
                    mtime = float(entry.stat().st_mtime)
                except Exception:
                    mtime = 0.0

                if base not in latest_by_base or mtime >= latest_mtime.get(base, -1.0):
                    latest_by_base[base] = entry.path
                    latest_mtime[base] = mtime
        except Exception:
            logging.debug("[bettercap.tool] failed to scan handshakes path: %s", handshakes_path, exc_info=True)
            return []

        return list(latest_by_base.values())

    def _latest_capture_file(self, handshakes_path: str, files: Optional[List[str]] = None) -> Optional[str]:
        files = files if isinstance(files, list) else self._list_unique_capture_files(handshakes_path)
        if not files:
            return None
        try:
            return max(files, key=lambda path: float(os.path.getmtime(path)))
        except Exception:
            return files[-1]

    def _resolve_capture_name(
        self,
        latest_capture: Optional[str] = None,
        latest_key: Optional[str] = None,
        bssid: Optional[str] = None,
    ) -> str:
        candidates: List[str] = []
        if bssid:
            candidates.append(str(bssid))
        if latest_key:
            candidates.append(str(latest_key))
        if latest_capture:
            candidates.append(self._capture_basename(latest_capture))

        mac_pat = re.compile(r'([0-9a-f]{2}(?::[0-9a-f]{2}){5})', re.IGNORECASE)
        for candidate in candidates:
            value = str(candidate or '').strip()
            if not value:
                continue

            if '->' in value:
                right = value.split('->')[-1].strip().lower()
                if right in self._mac_to_ssid:
                    return self._mac_to_ssid[right]

            lowered = value.lower()
            if lowered in self._mac_to_ssid:
                return self._mac_to_ssid[lowered]

            match = mac_pat.search(value)
            if match:
                mac = match.group(1).lower()
                if mac in self._mac_to_ssid:
                    return self._mac_to_ssid[mac]
                return mac

            cleaned = value.replace('_', ' ').replace('-', ' ').strip()
            if cleaned:
                return cleaned

        return "unknown"

    def _update_session_trophy(self, handshake_key: str) -> str:
        raw_key = str(handshake_key or '').strip()
        trophy_bssid = raw_key
        if '->' in raw_key:
            trophy_bssid = raw_key.split('->')[-1].strip()
        elif raw_key.startswith('ap:'):
            trophy_bssid = raw_key[3:].strip()

        trophy_name = self._resolve_capture_name(latest_key=raw_key, bssid=trophy_bssid)
        self._last_session_trophy_bssid = str(trophy_bssid or '').strip().lower()
        self._last_session_trophy_name = str(trophy_name or '').strip()
        logging.info(
            "[bettercap.tool] handshake trophy resolved: bssid=%s name=%s",
            self._last_session_trophy_bssid or 'unknown',
            self._last_session_trophy_name,
        )
        return self._last_session_trophy_name

    def _build_pwnd_widget(self, context: Dict[str, Any]) -> str:
        handshakes_path = str(self._resolve_value('handshakes_path', context) or '').strip()
        session_pwnd = len(self._known_handshakes)
        unique_captures = self._list_unique_capture_files(handshakes_path)
        total_pwnd = len(unique_captures)
        last_pwnd = str(self._last_session_trophy_name or '').strip() if session_pwnd > 0 else ''
        logging.info(
            "[bettercap.tool] pwnd widget updated: session=%d total=%d last=%r",
            session_pwnd,
            total_pwnd,
            last_pwnd,
        )
        if last_pwnd:
            return "%d (%d) %s" % (session_pwnd, total_pwnd, last_pwnd)
        return "%d (%d)" % (session_pwnd, total_pwnd)

    def _resolve_active_channel(
        self,
        context: Optional[Dict[str, Any]],
        preferred_channels: Optional[List[int]] = None,
        sess: Optional[Dict[str, Any]] = None,
    ) -> Optional[int]:
        context = context if isinstance(context, dict) else {}
        channel = self._normalize_channel(context.get('channel'))
        if channel is not None:
            self._active_channel = channel
            return channel

        if self._active_channel is not None:
            return self._active_channel

        sess = sess if isinstance(sess, dict) else {}
        for key in ('channel', 'current_channel', 'active_channel'):
            channel = self._normalize_channel(sess.get(key))
            if channel is not None:
                self._active_channel = channel
                return channel

        preferred_channels = preferred_channels if isinstance(preferred_channels, list) else []
        for candidate in preferred_channels:
            channel = self._normalize_channel(candidate)
            if channel is not None:
                return channel

        return None

    def _build_aps_widget(self, context: Dict[str, Any], env: Dict[str, Any]) -> str:
        active_channel = self._resolve_active_channel(
            context,
            preferred_channels=env.get('preferred_channels'),
            sess=env.get('session_wifi'),
        )
        total_aps = int(env.get('ap_count', 0.0) or 0.0)
        channel_ap_map = env.get('channel_ap_counts') if isinstance(env.get('channel_ap_counts'), dict) else {}
        channel_aps = int(channel_ap_map.get(active_channel, 0) or 0) if active_channel is not None else 0
        logging.info(
            "[bettercap.tool] aps widget updated: channel=%s channel_aps=%d total_aps=%d",
            active_channel if active_channel is not None else '?',
            channel_aps,
            total_aps,
        )
        if active_channel is None:
            return str(total_aps)
        return "%d (%d)" % (channel_aps, total_aps)

    def _voice_context(self, context: Dict[str, Any], env: Optional[Dict[str, Any]] = None, error: Optional[str] = None) -> Dict[str, Any]:
        payload = dict(context or {})
        env = env if isinstance(env, dict) else {}
        target_mac = str(payload.get('target_mac') or '').strip().lower()
        ap_mac = str(payload.get('ap_mac') or target_mac).strip().lower()
        ssid = self._mac_to_ssid.get(ap_mac) or self._mac_to_ssid.get(target_mac) or payload.get('ssid') or payload.get('hostname') or 'unknown target'
        payload.update({
            'ssid': ssid,
            'target_mac': target_mac or payload.get('target_mac') or '',
            'ap_mac': ap_mac or payload.get('ap_mac') or '',
            'channel': payload.get('channel') or (env.get('preferred_channels') or ['*'])[0],
            'client_count': int(env.get('client_count', 0.0) or 0.0),
            'reason': payload.get('reason') or error or '',
            'error': error or '',
        })
        return payload

    def _build_ui_event(self, event_name: str, context: Dict[str, Any], env: Optional[Dict[str, Any]] = None, error: Optional[str] = None) -> Dict[str, Any]:
        voice_context = self._voice_context(context, env=env, error=error)
        logging.debug(
            "[bettercap.tool] ui event: %s target=%s channel=%s error=%s",
            event_name,
            voice_context.get('target_mac', ''),
            voice_context.get('channel', ''),
            voice_context.get('error', ''),
        )
        return {
            'voice_event': event_name,
            'voice_context': voice_context,
        }

    def _ensure_client(self):
        """Ensures the API client is instantiated and running."""
        if self.client is None:
            logging.warning("[bettercap.tool] API client was not initialized, recreating it.")
            client_cfg = (self.client_manifest.get('client') or {})
            self.client = Client(
                hostname=str(client_cfg.get('hostname', 'localhost')),
                scheme=str(client_cfg.get('scheme', 'http')),
                port=int(client_cfg.get('port', 8081)),
                username=str(client_cfg.get('username', 'user')),
                password=str(client_cfg.get('password', 'pass')),
                request_timeout=int(client_cfg.get('request_timeout', 10)),
            )
        if hasattr(self.client, 'running'):
            self.client.running = True

    def on_load(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        self._ensure_client()
        if self.running:
            logging.warning("[bettercap.tool] start ignored, already running")
            return {'ok': True, 'already_running': True}
        context = context or {}
        self._runtime_defaults = self._resolve_runtime_defaults(context)
        self._api_failure_streak = 0
        self._deauth_window.clear()
        self._last_restart_signal_ts = 0.0
        self._last_connection_errors = int(getattr(self.client, '_connection_errors', 0) or 0)
        self._loaded = False
        self.running = True
        self._reset_session_pwnd('boot')
        return {'ok': True}

    def _lazy_load(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_client()
        self._runtime_defaults = self._resolve_runtime_defaults(context)
        self._api_failure_streak = 0
        self._deauth_window.clear()
        self._last_restart_signal_ts = 0.0
        self._last_connection_errors = int(getattr(self.client, '_connection_errors', 0) or 0)

        lifecycle = self.client_manifest.get('lifecycle') or {}
        load_spec = lifecycle.get('load') if isinstance(lifecycle, dict) else {}
        load_spec = load_spec if isinstance(load_spec, dict) else {}

        verify = load_spec.get('verify') if isinstance(load_spec, dict) else {}
        verify = verify if isinstance(verify, dict) else {}

        # API readiness + service bootstrap.
        readiness = self.client_manifest.get('readiness') or {}
        api_ready_cfg = verify.get('api_ready') if isinstance(verify.get('api_ready'), dict) else readiness
        api_enabled = bool(api_ready_cfg.get('enabled', bool(readiness.get('enabled', False))))
        if api_enabled and not self._ensure_api_ready(api_ready_cfg):
            self._loaded = False
            return {'ok': False, 'error': 'api_not_ready'}

        # Monitor interface assurance mirrors legacy startup behavior.
        mon_cfg = verify.get('monitor_interface') if isinstance(verify.get('monitor_interface'), dict) else {}
        if bool(mon_cfg.get('enabled', False)):
            iface = str(self._resolve_value('iface', context) or '')
            start_cmd_key = str(mon_cfg.get('start_cmd_key', 'mon_start_cmd'))
            start_cmd = self._resolve_value(start_cmd_key, context)
            ok = self._ensure_monitor_interface(
                iface=iface,
                mon_start_cmd=str(start_cmd) if start_cmd else None,
                timeout=int(mon_cfg.get('timeout', 60)),
                interval=float(mon_cfg.get('interval', 1)),
            )
            if not ok:
                self._loaded = False
                return {'ok': False, 'error': 'monitor_interface_unavailable'}

        commands = load_spec.get('commands') if isinstance(load_spec, dict) else None
        if not commands:
            startup = self.client_manifest.get('startup') or {}
            if bool(startup.get('auto_start', False)):
                commands = startup.get('commands') or []

        cmd_result = self._run_command_list(commands or [], context=context)
        self._loaded = bool(cmd_result.get('ok', True))
        return cmd_result

    def on_unload(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        context = context or {}
        if self.client and hasattr(self.client, 'running'):
            self.client.running = False

        lifecycle = self.client_manifest.get('lifecycle') or {}
        unload_spec = lifecycle.get('unload') if isinstance(lifecycle, dict) else {}
        unload_spec = unload_spec if isinstance(unload_spec, dict) else {}

        commands = unload_spec.get('commands') or []
        result = self._run_command_list(commands, context=context)

        service_cfg = self._service_cfg('stop')
        if bool(service_cfg.get('enabled', False)):
            self._run_shell_commands(service_cfg.get('commands') or [], strict=False)

        self._loaded = False
        self.running = False
        self._runtime_defaults = {}
        self._reset_session_pwnd('unload')
        logging.info("[bettercap.tool] unloaded (runtime destroyed)")
        result['events'] = self._build_ui_event('tool_unloaded', context)
        return result

    def on_pause(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Lightweight pause, keeps services and clients alive but stops polling."""
        logging.info("[bettercap.tool] paused (runtime preserved)")
        self.running = False
        return {'ok': True, 'events': self._build_ui_event('tool_paused', context or {})}

    def on_resume(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Lightweight resume, assumes services and clients are alive."""
        logging.info("[bettercap.tool] resumed (runtime preserved)")
        self.running = True
        return {'ok': True, 'events': self._build_ui_event('tool_resumed', context or {})}

    def on_restart(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        self._ensure_client()
        context = context or {}
        self._api_failure_streak = 0
        self._deauth_window.clear()
        self._last_restart_signal_ts = 0.0
        self._last_connection_errors = int(getattr(self.client, '_connection_errors', 0) or 0)
        self._last_recon_refresh_ts = time.time()
        lifecycle = self.client_manifest.get('lifecycle') or {}
        restart_spec = lifecycle.get('restart') if isinstance(lifecycle, dict) else {}
        restart_spec = restart_spec if isinstance(restart_spec, dict) else {}

        service_cfg = self._service_cfg('restart')
        if bool(service_cfg.get('enabled', False)):
            shell_res = self._run_shell_commands(service_cfg.get('commands') or [], strict=False)
            if not shell_res.get('ok', True):
                return {'ok': False, 'error': 'service_restart_failed'}
            api_cfg = (self.client_manifest.get('readiness') or {})
            if not self._ensure_api_ready(api_cfg):
                return {'ok': False, 'error': 'api_not_ready'}

        commands = restart_spec.get('commands') or []
        if commands:
            self._runtime_defaults = self._resolve_runtime_defaults(context)
            return self._run_command_list(commands, context=context)

        down = self.on_unload(context=context)
        if not down.get('ok', False):
            return down
        return self.on_load(context=context)

    def execute(self, command_id: str, context: Dict[str, Any], command_spec: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_client()
        context = context or {}
        env = self._collect_environment_metrics(context=context)
        self._enrich_context_from_environment(context, command_id, env)

        # Auto-populate target context from live Bettercap state when possible.
        if not context.get('target_mac'):
            target_mac = self._pick_target_for_command(command_id)
            if target_mac:
                context['target_mac'] = target_mac

        if not self.running:
            logging.warning("[bettercap.tool] tool not running, ignoring action: %s", command_id)
            return {'ok': False, 'error': 'tool_stopped', 'signals': {}, 'metrics': {}, 'costs': {}}

        if not self._loaded:
            load_result = self._lazy_load(context)
            if not load_result.get('ok', False):
                return {
                    'ok': False,
                    'error': load_result.get('error', 'tool_lazy_load_failed'),
                    'signals': {
                        'resource_unavailable': 1.0,
                        'execution_timeout': 0.0,
                        'invalid_target': 0.0,
                        'interaction_success': 0.0,
                    },
                    'metrics': {},
                }

        required = command_spec.get('required_context') or []
        missing = [k for k in required if self._resolve_value(k, context) is None]
        if missing:
            return {
                'ok': False,
                'error': 'invalid_target',
                'signals': {
                    'resource_unavailable': 0.0,
                    'execution_timeout': 0.0,
                    'invalid_target': 1.0,
                    'interaction_success': 0.0,
                },
                'metrics': {},
            }

        runs = command_spec.get('runs') or []
        if not runs and command_spec.get('run'):
            runs = [command_spec['run']]

        ok = True
        error = None
        responses: List[Any] = []
        injection_error_count = 0
        timeout_error_count = 0
        deauth_requested = 0
        deauth_attempted = False

        for cmd_tpl in runs:
            cmd = self._format_cmd(str(cmd_tpl), context)
            if cmd is None:
                ok = False
                error = 'invalid_target'
                break
            if cmd.strip().lower().startswith('wifi.deauth '):
                deauth_requested += 1
                deauth_attempted = True

            cmd_started = time.time()
            logging.info("[bettercap.tool] run: %s", cmd)
            res = self.client.run(cmd, verbose_errors=False)
            logging.info(
                "[bettercap.tool] run done: ok=%s elapsed=%.3fs",
                bool(res is not None),
                max(time.time() - cmd_started, 0.0),
            )
            responses.append(res)
            if res is None:
                ok = False
                error = 'execution_timeout'
                timeout_error_count += 1
                break
            if isinstance(res, str):
                low = res.lower()
                if (
                    'unknown bssid' in low
                    or 'deauth skip list' in low
                    or "doesn't have detected clients" in low
                    or 'invalid bssid' in low
                ):
                    ok = False
                    error = 'invalid_target'
                    break
                if 'resource temporarily unavailable' in low or 'could not inject wifi packet' in low:
                    ok = False
                    error = 'resource_unavailable'
                    injection_error_count += 1
                    break
                if low.startswith('error '):
                    ok = False
                    error = 'resource_unavailable'
                    break

        if error == 'resource_unavailable' and injection_error_count == 0:
            injection_error_count = 1
        if error == 'execution_timeout' and timeout_error_count == 0:
            timeout_error_count = 1

        runtime = self._sample_runtime_events()
        injection_error_count += int(runtime.get('injection_errors', 0))
        runtime_handshakes = runtime.get('handshake_keys', [])
        runtime_deauth_events = int(runtime.get('deauth_events', 0))

        # Reflect asynchronous injection failures as action pressure/failure.
        if ok and injection_error_count > 0:
            ok = False
            error = 'resource_unavailable'

        # If we requested deauth but no deauth event appears, treat as ineffective.
        if ok and deauth_requested > 0 and runtime_deauth_events <= 0:
            ok = False
            error = 'ineffective_action'
            logging.warning("[bettercap.tool] deauth command did not produce deauth event (target may be invalid/ineligible)")

        if not env.get('session_ok', False) and error is None:
            error = 'resource_unavailable'
            ok = False

        connection_errors = int(getattr(self.client, '_connection_errors', 0) or 0)
        connection_delta = max(connection_errors - self._last_connection_errors, 0)
        self._last_connection_errors = connection_errors
        if (not ok and error in ('execution_timeout', 'resource_unavailable')) or connection_delta > 0 or not env.get('session_ok', True):
            self._api_failure_streak += 1
        else:
            self._api_failure_streak = 0

        recovery_cfg = self.tool_manifest.get('recovery', {}) if isinstance(self.tool_manifest.get('recovery', {}), dict) else {}
        deauth_window_min_attempts = int(recovery_cfg.get('deauth_window_min_attempts', 6) or 6)
        deauth_window_size = int(recovery_cfg.get('deauth_window_size', 20) or 20)
        deauth_failure_ratio_threshold = float(recovery_cfg.get('deauth_failure_ratio_threshold', 0.8) or 0.8)
        restart_signal_cooldown = float(recovery_cfg.get('restart_signal_cooldown_seconds', 120.0) or 120.0)
        injection_restart_threshold = int(recovery_cfg.get('injection_restart_threshold', 10) or 10)

        if self._deauth_window.maxlen != max(deauth_window_size, 1):
            self._deauth_window = deque(self._deauth_window, maxlen=max(deauth_window_size, 1))

        if deauth_attempted:
            self._deauth_window.append(1 if error == 'ineffective_action' else 0)

        deauth_attempts = len(self._deauth_window)
        deauth_failures = sum(self._deauth_window)
        deauth_failure_ratio = (float(deauth_failures) / float(deauth_attempts)) if deauth_attempts > 0 else 0.0

        restart_due_to_deauth = (
            deauth_attempts >= max(deauth_window_min_attempts, 1)
            and deauth_failure_ratio > deauth_failure_ratio_threshold
        )

        now = time.time()
        restart_recommended = (
            self._api_failure_streak >= 3
            or injection_error_count >= max(injection_restart_threshold, 1)
            or env.get('recovery_recommended', False)
        )
        if restart_due_to_deauth and (now - self._last_restart_signal_ts) > restart_signal_cooldown:
            restart_recommended = True
            self._last_restart_signal_ts = now

        signals = {
            'interaction_success': 1.0 if ok else 0.0,
            'execution_timeout': 1.0 if error == 'execution_timeout' else 0.0,
            'resource_unavailable': 1.0 if error == 'resource_unavailable' else 0.0,
            'invalid_target': 1.0 if error == 'invalid_target' else 0.0,
            'interaction_failures': 0.0 if ok else 1.0,
            'timeout_errors': float(timeout_error_count),
            'injection_errors': float(injection_error_count),
            'bettercap_errors': 1.0 if error in ('resource_unavailable', 'execution_timeout') else 0.0,
            'module_instability': 1.0 if error in ('resource_unavailable', 'execution_timeout') else 0.0,
            'ineffective_action': 1.0 if error == 'ineffective_action' else 0.0,
            'ap_discovery_event': float(env.get('ap_delta', 0) > 0),
            'client_discovery_event': float(env.get('client_delta', 0) > 0),
            'recon_stall': float(env.get('recon_stall', 0.0)),
            'restart_recommended': float(restart_recommended),
            'deauth_failure_ratio': float(deauth_failure_ratio),
            'deauth_failure_attempts': float(deauth_attempts),
        }

        new_handshakes = self._detect_new_handshakes()
        for hs in runtime_handshakes:
            if hs not in new_handshakes:
                new_handshakes.append(hs)
        if new_handshakes:
            for hs in new_handshakes:
                logging.warning("!!! [kali] captured new handshake: %s !!!", hs)
            signals['handshake_event'] = 1.0

        metrics = {
            'targets_visible': float(env.get('targets_visible', 0.0)),
            'ap_count': float(env.get('ap_count', 0.0)),
            'client_count': float(env.get('client_count', 0.0)),
            'environment_activity': float(env.get('environment_activity', 0.0)),
            'channel_density': float(env.get('channel_density', 0.0)),
            'client_reappearance_rate': float(env.get('client_reappearance_rate', 0.0)),
            'ap_stability_score': float(env.get('ap_stability_score', 0.0)),
            'blind_for_epochs': float(1.0 if env.get('ap_count', 0.0) <= 0.0 else 0.0),
            'new_handshakes': float(len(new_handshakes)),
        }

        # Construct explicit UI events for the VoiceWidgetManager.
        events = {
            'widgets': {
                'channel': self._normalize_channel(context.get('channel')) or (env.get('preferred_channels')[0] if env.get('preferred_channels') else '*'),
                'aps': self._build_aps_widget(context, env),
                'pwnd': self._build_pwnd_widget(context),
            }
        }

        explicit_event = None
        if error == 'resource_unavailable':
            explicit_event = 'resource_unavailable'
        elif error == 'execution_timeout':
            explicit_event = 'timeout'
        elif error == 'ineffective_action':
            explicit_event = 'deauth_ineffective'
        elif restart_recommended:
            explicit_event = 'radio_recovering'
        elif ok:
            explicit_event = {
                'interact_with_target': 'assoc_sent',
                'force_state_change': 'deauth_sent',
                'set_channel': 'channel_locked',
                'focus_context': 'channel_scan_focus',
                'explore_environment': 'recon_started',
                'start_recon': 'recon_started',
                'stop_recon': 'recon_stopped',
                'restart_recon': 'recon_restarted',
                'clear_state': 'state_cleared',
                'sync_events': 'events_synced',
            }.get(command_id)

        if new_handshakes:
            raw_hs = new_handshakes[-1]
            trophy_name = self._update_session_trophy(raw_hs)
            target_mac = self._last_session_trophy_bssid or raw_hs
            events['widgets']['pwnd'] = self._build_pwnd_widget(context)
            events.update(self._build_ui_event('handshake_captured', {'ssid': trophy_name, 'target_mac': target_mac}, env=env))
        elif env.get('ap_delta', 0) > 0:
            events.update(self._build_ui_event('target_spotted', {'ssid': 'nearby network'}, env=env))
        elif explicit_event:
            events.update(self._build_ui_event(explicit_event, context, env=env, error=error))

        result = {
            'ok': ok,
            'error': error,
            'signals': signals,
            'metrics': metrics,
            'responses': responses,
            'events': events,
        }
        if ok and command_id in ('start_recon', 'restart_recon', 'explore_environment', 'focus_context', 'set_channel'):
            self._last_recon_refresh_ts = time.time()
        if 'ui' in command_spec:
            result['ui'] = command_spec['ui']
        return result

    def _enrich_context_from_environment(self, context: Dict[str, Any], command_id: str, env: Dict[str, Any]):
        if command_id in ('focus_context', 'explore_environment') and not context.get('channels_csv'):
            channels = env.get('preferred_channels') or []
            if channels:
                context['channels_csv'] = ','.join(str(ch) for ch in channels)
        if command_id == 'set_channel' and not context.get('channel'):
            channels = env.get('preferred_channels') or []
            if channels:
                context['channel'] = channels[0]

    def _collect_environment_metrics(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.client:
            return {
                'session_ok': False,
                'targets_visible': 0.0,
                'ap_count': 0.0,
                'client_count': 0.0,
                'environment_activity': 0.0,
                'channel_density': 0.0,
                'client_reappearance_rate': 0.0,
                'ap_stability_score': 0.0,
                'preferred_channels': [],
                'channel_ap_counts': {},
                'active_channel': None,
                'session_wifi': {},
                'ap_delta': 0,
                'client_delta': 0,
                'recon_stall': 1.0,
                'recovery_recommended': True,
            }
        sess = self.client.session('session/wifi')
        if not isinstance(sess, dict):
            return {
                'session_ok': False,
                'targets_visible': 0.0,
                'ap_count': 0.0,
                'client_count': 0.0,
                'environment_activity': 0.0,
                'channel_density': 0.0,
                'client_reappearance_rate': 0.0,
                'ap_stability_score': 0.0,
                'preferred_channels': [],
                'channel_ap_counts': {},
                'active_channel': None,
                'session_wifi': {},
                'ap_delta': 0,
                'client_delta': 0,
                'recon_stall': 1.0,
                'recovery_recommended': True,
            }

        aps = sess.get('aps') or []
        now = time.time()
        ap_ids: Dict[str, float] = {}
        client_ids: Dict[str, float] = {}
        channel_activity: Dict[int, float] = {}
        channel_ap_counts: Dict[int, int] = {}
        active_ap_count = 0
        total_packets = 0.0
        stable_ap_count = 0
        reappeared_clients = 0

        for ap in aps:
            if not isinstance(ap, dict):
                continue

            ap_mac = str(ap.get('mac') or ap.get('bssid') or '').lower()
            if not ap_mac:
                continue

            # Map BSSID to human-readable SSID
            ssid = str(ap.get('hostname') or ap.get('ssid') or ap.get('essid') or 'hidden').strip()
            if ssid:
                self._mac_to_ssid[ap_mac] = ssid

            packets = self._safe_float(ap.get('packets'))
            if packets <= 0.0:
                packets = self._safe_float(ap.get('sent')) + self._safe_float(ap.get('received'))
            total_packets += max(packets, 0.0)

            clients = ap.get('clients') or []
            if isinstance(clients, list) and clients:
                active_ap_count += 1

            last_seen = self._safe_float(ap.get('last_seen'))
            if last_seen <= 0.0:
                last_seen = now
            ap_ids[ap_mac] = last_seen

            previous_seen = self._safe_float(self._last_wifi_snapshot.get('aps', {}).get(ap_mac))
            if previous_seen > 0.0 and (now - previous_seen) <= 90.0:
                stable_ap_count += 1

            channel = self._normalize_channel(ap.get('channel'))
            if channel is not None:
                channel_ap_counts[channel] = channel_ap_counts.get(channel, 0) + 1
                channel_activity[channel] = channel_activity.get(channel, 0.0) + 1.0 + float(len(clients))

            for sta in clients:
                if not isinstance(sta, dict):
                    continue
                sta_mac = str(sta.get('mac') or '').lower()
                if not sta_mac or sta_mac == ap_mac:
                    continue
                sta_seen = self._safe_float(sta.get('last_seen'))
                if sta_seen <= 0.0:
                    sta_seen = last_seen
                client_ids[sta_mac] = sta_seen

                previous_client_seen = self._safe_float(self._last_wifi_snapshot.get('clients', {}).get(sta_mac))
                if previous_client_seen > 0.0 and (now - previous_client_seen) > 15.0:
                    reappeared_clients += 1

        ap_count = len(ap_ids)
        client_count = len(client_ids)
        targets_visible = active_ap_count

        ap_delta = max(ap_count - len(self._last_wifi_snapshot.get('aps', {})), 0)
        client_delta = max(client_count - len(self._last_wifi_snapshot.get('clients', {})), 0)
        if ap_delta > 0 or client_delta > 0 or targets_visible > 0:
            self._last_discovery_ts = now

        channel_values = list(channel_activity.values())
        if channel_values:
            self._channel_history.append(sum(channel_values) / len(channel_values))
        channel_density = (sum(channel_values) / len(channel_values)) if channel_values else 0.0

        denom_clients = max(client_count, 1)
        denom_aps = max(ap_count, 1)
        environment_activity = min((client_count + active_ap_count + min(total_packets / 250.0, 10.0)) / 10.0, 1.0)
        client_reappearance_rate = min(reappeared_clients / float(denom_clients), 1.0)
        ap_stability_score = min(stable_ap_count / float(denom_aps), 1.0)

        recon_stall_threshold = 45.0
        recon_stall = 1.0 if (ap_count <= 0 and (now - self._last_discovery_ts) >= recon_stall_threshold) else 0.0
        preferred_channels = [
            channel for channel, _ in sorted(channel_activity.items(), key=lambda item: item[1], reverse=True)[:3]
        ]
        if not preferred_channels:
            preferred_channels = [1, 6, 11]
        active_channel = self._resolve_active_channel(context, preferred_channels=preferred_channels, sess=sess)

        self._last_wifi_snapshot = {
            'aps': ap_ids,
            'clients': client_ids,
            'channel_activity': channel_activity,
        }

        return {
            'session_ok': True,
            'targets_visible': float(targets_visible),
            'ap_count': float(ap_count),
            'client_count': float(client_count),
            'environment_activity': float(environment_activity),
            'channel_density': float(min(channel_density / 6.0, 1.0)),
            'client_reappearance_rate': float(client_reappearance_rate),
            'ap_stability_score': float(ap_stability_score),
            'preferred_channels': preferred_channels,
            'channel_ap_counts': channel_ap_counts,
            'active_channel': active_channel,
            'session_wifi': sess,
            'ap_delta': ap_delta,
            'client_delta': client_delta,
            'recon_stall': recon_stall,
            'recovery_recommended': bool(recon_stall and (now - self._last_recon_refresh_ts) >= 60.0),
        }

    def _sample_runtime_events(self) -> Dict[str, Any]:
        now = time.time()
        since_ts = max(self._last_journal_probe_ts, now - 20.0)
        self._last_journal_probe_ts = now

        # Use system journal as a fallback for async Bettercap errors/events that
        # don't show up in direct command responses.
        cmd = 'journalctl --no-pager -S @%d -o cat' % int(since_ts)
        try:
            proc = subprocess.run(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=2,
            )
            if proc.returncode != 0 and not proc.stdout:
                return {'injection_errors': 0, 'deauth_events': 0, 'handshake_keys': []}
            text = proc.stdout or ''
        except Exception:
            return {'injection_errors': 0, 'deauth_events': 0, 'handshake_keys': []}

        inj_pat = re.compile(r"wifi could not inject WiFi packet: send: Resource temporarily unavailable", re.IGNORECASE)
        deauth_pat = re.compile(r"wifi deauthing client ", re.IGNORECASE)
        hs_pat_a = re.compile(
            r"wifi\.client\.handshake.*captured\s+([0-9a-f]{2}(?::[0-9a-f]{2}){5}).*?\(([0-9a-f]{2}(?::[0-9a-f]{2}){5})\)",
            re.IGNORECASE,
        )
        hs_pat_b = re.compile(r"captured .*handshake.*?([0-9a-f]{2}(?::[0-9a-f]{2}){5})", re.IGNORECASE)

        injection_errors = len(inj_pat.findall(text))
        deauth_events = len(deauth_pat.findall(text))

        handshake_keys: List[str] = []
        for match in hs_pat_a.finditer(text):
            sta = (match.group(1) or '').lower()
            ap = (match.group(2) or '').lower()
            key = ("%s -> %s" % (sta, ap)).strip()
            if not key or key in self._seen_journal_handshakes:
                continue
            self._seen_journal_handshakes.add(key)
            handshake_keys.append(key)

        for match in hs_pat_b.finditer(text):
            mac = (match.group(1) or '').lower()
            if not mac:
                continue
            self._seen_journal_handshakes.add(mac)
            if mac not in handshake_keys:
                handshake_keys.append(mac)

        return {
            'injection_errors': injection_errors,
            'deauth_events': deauth_events,
            'handshake_keys': handshake_keys,
        }

    def _handshake_keys_from_session(self, sess: Dict[str, Any]) -> Set[str]:
        keys: Set[str] = set()
        if not isinstance(sess, dict):
            return keys

        aps = sess.get('aps') or []
        for ap in aps:
            if not isinstance(ap, dict):
                continue

            ap_mac = str(ap.get('mac') or ap.get('bssid') or 'unknown-ap')

            # Some Bettercap builds expose per-AP handshake booleans/metadata.
            if ap.get('handshake'):
                keys.add("ap:%s" % ap_mac)

            handshakes = ap.get('handshakes') or []
            if isinstance(handshakes, list):
                for hs in handshakes:
                    if isinstance(hs, dict):
                        sta = str(hs.get('station') or hs.get('client') or hs.get('mac') or 'unknown-sta')
                        keys.add("%s->%s" % (sta, ap_mac))
                    elif isinstance(hs, str):
                        keys.add("%s@%s" % (hs, ap_mac))

            for sta in (ap.get('clients') or []):
                if not isinstance(sta, dict):
                    continue
                if sta.get('handshake') or sta.get('captured_handshake'):
                    sta_mac = str(sta.get('mac') or 'unknown-sta')
                    keys.add("%s->%s" % (sta_mac, ap_mac))

        top_hs = sess.get('handshakes')
        if isinstance(top_hs, list):
            for hs in top_hs:
                if isinstance(hs, dict):
                    ap_mac = str(hs.get('ap') or hs.get('bssid') or 'unknown-ap')
                    sta_mac = str(hs.get('station') or hs.get('client') or hs.get('mac') or 'unknown-sta')
                    keys.add("%s->%s" % (sta_mac, ap_mac))
                elif isinstance(hs, str):
                    keys.add(hs)

        return keys

    def _detect_new_handshakes(self) -> List[str]:
        if not self.client:
            return []
        try:
            sess = self.client.session('session/wifi')
            keys = self._handshake_keys_from_session(sess if isinstance(sess, dict) else {})
            if not self._known_handshakes:
                self._known_handshakes = set(keys)
                return []

            new_keys = sorted(k for k in keys if k not in self._known_handshakes)
            if new_keys:
                self._known_handshakes.update(new_keys)
            return new_keys
        except Exception:
            return []

    def _pick_target_for_command(self, command_id: str):
        # Deauth is far more reliable when targeting a station/client MAC.
        if command_id == 'force_state_change':
            return self._pick_client_mac()
        return self._pick_ap_mac()

    def _pick_client_mac(self):
        if not self.client:
            return None
        try:
            sess = self.client.session('session/wifi')
            if not isinstance(sess, dict):
                return None

            aps = sess.get('aps') or []
            if not aps:
                return None

            candidates = []
            for ap in aps:
                if not isinstance(ap, dict):
                    continue
                ap_rssi = float(ap.get('rssi') or -1000.0)
                for sta in (ap.get('clients') or []):
                    if not isinstance(sta, dict):
                        continue
                    mac = sta.get('mac')
                    if not mac:
                        continue
                    # Bettercap sometimes reports AP MAC as a "client"; deauthing that is ineffective.
                    if str(mac).lower() == str(ap.get('mac') or '').lower():
                        continue
                    sta_rssi = float(sta.get('rssi') or ap_rssi)
                    candidates.append({'mac': str(mac), 'rssi': sta_rssi})

            if not candidates:
                return None

            ranked = sorted(candidates, key=lambda c: c['rssi'], reverse=True)
            start = self._target_cursor % len(ranked)
            for offs in range(len(ranked)):
                c = ranked[(start + offs) % len(ranked)]
                if c.get('mac'):
                    self._target_cursor = (start + offs + 1) % len(ranked)
                    return c['mac']
        except Exception:
            return None
        return None

    def _pick_ap_mac(self):
        if not self.client:
            return None
        try:
            sess = self.client.session('session/wifi')
            if not isinstance(sess, dict):
                return None

            aps = sess.get('aps') or []
            if not aps:
                return None

            # Prefer APs with clients and stronger signal, then rotate to avoid action spam on one AP.
            ranked = sorted(
                aps,
                key=lambda ap: (
                    len(ap.get('clients') or []),
                    float(ap.get('rssi') or -1000.0),
                ),
                reverse=True,
            )

            if not ranked:
                return None

            start = self._target_cursor % len(ranked)
            for offs in range(len(ranked)):
                ap = ranked[(start + offs) % len(ranked)]
                mac = ap.get('mac')
                if mac:
                    self._target_cursor = (start + offs + 1) % len(ranked)
                    return str(mac)
        except Exception:
            return None
        return None

    def _service_cfg(self, key: str) -> Dict[str, Any]:
        service = self.client_manifest.get('service') or {}
        cfg = service.get(key) if isinstance(service, dict) else {}
        return cfg if isinstance(cfg, dict) else {}

    def _ensure_api_ready(self, api_ready_cfg: Dict[str, Any]) -> bool:
        self._ensure_client()
        if not self.client:
            logging.error("[bettercap.tool] cannot ensure API readiness, client is not initialized.")
            return False

        timeout = int(api_ready_cfg.get('timeout', 45))
        interval = int(api_ready_cfg.get('interval', 2))

        start_cfg = self._service_cfg('start')
        start_enabled = bool(start_cfg.get('enabled', False))
        start_commands = start_cfg.get('commands') or []

        # Use a short, quiet probe first to see if the API is already available.
        if self.client._probe_ready():
            logging.info("[bettercap.tool] API was already ready.")
            return True

        logging.info("[bettercap.tool] API is not ready, attempting to start service...")

        if not start_enabled or not start_commands:
            logging.warning("[bettercap.tool] Service start is not enabled or no start commands are configured.")
            # Fallback to waiting for the full duration in case the service is just starting slowly on its own.
            return self.client.wait_until_ready(timeout=timeout, interval=interval)

        # First attempt to start the service.
        self._run_shell_commands(start_commands, strict=False)

        # Now, wait for the full duration for the API to become ready.
        if self.client.wait_until_ready(timeout=timeout, interval=interval):
            return True

        # If it's still not ready, enter the retry loop based on manifest configuration.
        retries = int(start_cfg.get('retries', 1))
        if retries <= 0:
            logging.error("[bettercap.tool] Failed to bring API up.")
            return False

        wait_timeout = int(start_cfg.get('wait_timeout', timeout))

        logging.info("[bettercap.tool] API still not ready, entering retry loop (%d retries)...", retries)
        for i in range(retries):
            logging.info("[bettercap.tool] Service start retry #%d of %d...", i + 1, retries)
            self._run_shell_commands(start_commands, strict=False)
            if self.client.wait_until_ready(timeout=wait_timeout, interval=interval):
                return True

        logging.error("[bettercap.tool] Failed to bring API up after all retries.")
        return False

    def _run_shell_commands(self, commands: List[str], strict: bool = False) -> Dict[str, Any]:
        last_rc = 0
        for cmd in commands:
            proc = subprocess.run(
                str(cmd),
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=15
            )
            last_rc = proc.returncode
            if proc.returncode != 0:
                logging.warning("[bettercap.tool] service command failed (rc=%s): %s", proc.returncode, cmd)
                if proc.stderr:
                    logging.warning("[bettercap.tool] service command stderr: %s", proc.stderr.strip())
                if strict:
                    return {'ok': False, 'returncode': proc.returncode}
        return {'ok': last_rc == 0 if commands else True, 'returncode': last_rc}

    def _is_service_active(self, check_cmd: str) -> bool:
        """
        Runs a shell command to determine if the bettercap service is active.
        Returns True if the command exits with return code 0.
        """
        if not check_cmd:
            return False

        try:
            proc = subprocess.run(
                check_cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5
            )

            if proc.returncode == 0:
                return True

            logging.debug(
                "[bettercap.tool] service check failed (rc=%s): %s",
                proc.returncode,
                check_cmd
            )
            return False

        except Exception as exc:
            logging.warning(
                "[bettercap.tool] service check exception: %s",
                exc
            )
            return False

    def shutdown(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        logging.info("[bettercap.tool] shutting down runtime")
        self.on_unload(context)
        if self.client:
            try:
                # close() sets client.running = False, stopping the websocket poller.
                self.client.close()
            except Exception as exc:
                logging.warning("[bettercap.tool] exception while closing client: %s", exc)
        return {'ok': True}

    def _run_command_list(self, commands: List[str], context: Dict[str, Any]) -> Dict[str, Any]:
        responses: List[Any] = []
        for cmd_tpl in commands:
            cmd = self._format_cmd(str(cmd_tpl), context)
            if not cmd:
                return {'ok': False, 'error': 'invalid_target', 'responses': responses}
            res = self.client.run(cmd, verbose_errors=False)
            responses.append(res)
            if res is None:
                return {'ok': False, 'error': 'execution_timeout', 'responses': responses}
        return {'ok': True, 'responses': responses}

    def _resolve_runtime_defaults(self, context: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = dict(self.defaults)

        config = context.get('config') if isinstance(context.get('config'), dict) else {}
        main = config.get('main') if isinstance(config.get('main'), dict) else {}
        personality = config.get('personality') if isinstance(config.get('personality'), dict) else {}
        bettercap = config.get('bettercap') if isinstance(config.get('bettercap'), dict) else {}

        mappings = {
            'iface': main.get('iface'),
            'ap_ttl': personality.get('ap_ttl'),
            'sta_ttl': personality.get('sta_ttl'),
            'min_rssi': personality.get('min_rssi'),
            'handshakes_path': bettercap.get('handshakes'),
            'mon_start_cmd': main.get('mon_start_cmd'),
        }

        for key, value in mappings.items():
            if value is not None:
                out[key] = value

        for key, value in context.items():
            if value is not None:
                out[key] = value

        return out

    def _ensure_monitor_interface(self, iface: str, mon_start_cmd: str = None, timeout: int = 60, interval: float = 1.0) -> bool:
        if not self.client:
            logging.error("[bettercap.tool] cannot ensure monitor interface, client is not initialized.")
            return False
        if not iface:
            return False

        end_ts = time.time() + max(timeout, 1)
        last_start = 0.0

        while time.time() < end_ts:
            sess = self.client.session()
            if isinstance(sess, dict):
                interfaces = sess.get('interfaces') or []
                for ifc in interfaces:
                    if isinstance(ifc, dict) and ifc.get('name') == iface:
                        return True

            # Try to bring monitor mode up (throttled)
            if mon_start_cmd and (time.time() - last_start) >= 3.0:
                self.client.run(f"!{mon_start_cmd}", verbose_errors=False)
                last_start = time.time()

            time.sleep(max(interval, 0.2))

        return False

    def _fetch_targets_visible(self) -> int:
        env = self._collect_environment_metrics()
        return int(env.get('targets_visible', 0.0) or 0.0)

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _normalize_channel(value: Any) -> Optional[int]:
        try:
            ch = int(value)
        except (TypeError, ValueError):
            return None
        return ch if ch > 0 else None

    def _resolve_value(self, key: str, context: Dict[str, Any]):
        if key in context:
            return context[key]

        state = context.get('state') if isinstance(context.get('state'), dict) else {}
        if key in state:
            return state[key]

        if key in self._runtime_defaults:
            return self._runtime_defaults.get(key)

        return self.defaults.get(key)

    def _format_cmd(self, template: str, context: Dict[str, Any]):
        keys = [fname for _, fname, _, _ in string.Formatter().parse(template) if fname]
        values = {}
        for key in keys:
            val = self._resolve_value(key, context)
            if val is None:
                return None
            values[key] = val

        try:
            return template.format(**values)
        except Exception:
            return None

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        if yaml is None:
            return {}

        path = os.path.join(self.manifest_dir, filename)
        if not os.path.isfile(path):
            return {}

        with open(path, 'rt', encoding='utf-8') as fp:
            data = yaml.safe_load(fp) or {}
            return data if isinstance(data, dict) else {}
