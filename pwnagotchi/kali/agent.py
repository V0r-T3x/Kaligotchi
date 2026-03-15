import json
import logging
import os
import re
import asyncio
import threading
import time

import pwnagotchi
import pwnagotchi.plugins as plugins
import pwnagotchi.utils as utils
import pwnagotchi.ui.faces as faces
from pwnagotchi.log import LastSession
from pwnagotchi.mesh.utils import AsyncAdvertiser
from pwnagotchi.ui.web.server import Server

from pwnagotchi.kali.ai.reflex import ReflexBrain
import pwnagotchi.kali.ai as ai
from pwnagotchi.kali.ai.train import AsyncTrainer
from pwnagotchi.kali.automata import Automata
from pwnagotchi.kali.toolbox import ManifestToolManager

RECOVERY_DATA_FILE = '/root/.pwnagotchi-recovery'
DEFAULT_KALI_BRAINS_ROOT = '/root/brains'
TOOL_STATE_FILE = '/root/.kali-tool-state.json'


class Agent(Automata, AsyncAdvertiser, AsyncTrainer):
    STATE_RUNNING = 'running'
    STATE_IDLE = 'idle'
    STATE_STOPPED = 'stopped'

    def __init__(self, view, config, keypair):
        Automata.__init__(self, config, view)
        AsyncAdvertiser.__init__(self, config, view, keypair)
        AsyncTrainer.__init__(self, config)

        self.client = None
        self._reflex = None

        self._started_at = time.time()
        self._kali_session_started_at = time.time()
        self._filter = None if not config['main']['filter'] else re.compile(config['main']['filter'])
        self._supported_channels = utils.iface_channels(config['main']['iface'])
        self._allowed_channels = utils.iface_channels(config['main']['iface'], disabled=False)
        self._view = view
        self._view.set_agent(self)
        self._web_ui = Server(self, config['ui'])

        self.last_session = LastSession(self._config)
        self.current_session = LastSession(self._config)
        self.mode = 'kali'

        toolbox_root = os.path.join(os.path.dirname(__file__), 'toolbox')
        self._toolbox = ManifestToolManager(toolbox_root)
        self._toolbox.bind_view(self._view)

        self._all_tool_actions = []
        self._tool_actions = []
        self._active_tool_id = None
        self._tool_brains = {}
        self._ai_pause = False
        self._runtime_mode = 'stopped'

        self._auto_thread = None
        self._auto_stop = threading.Event()
        self._auto_action_ptr = 0
        self._event_thread = None
        self._event_stop = threading.Event()
        self._last_injection_warn = 0.0
        self._started = False
        self.state = self.STATE_IDLE

        logging.info("%s@%s (v%s)", pwnagotchi.name(), self.fingerprint(), pwnagotchi.__version__)
        for _, plugin in plugins.loaded.items():
            logging.debug("plugin '%s' v%s", plugin.__class__.__name__, plugin.__version__)

    def config(self):
        return self._config

    def supported_channels(self):
        return self._supported_channels

    def tools(self):
        return self._toolbox.list_tools()

    def active_tool(self):
        return self._active_tool_id
    
    def active_tool_name(self):
        # Delegate to toolbox as the source of truth for active tool.
        if hasattr(self, '_toolbox') and self._toolbox:
            return self._toolbox.active_tool() or 'none'
        return self._active_tool_id or 'none'

    def runtime_state(self):
        return str(self._runtime_mode or 'stopped')

    def _set_bootstrap_status(self, text, face=None):
        message = str(text or '').strip()
        if not message:
            return
        logging.info("[kali] %s", message)
        if not self._view:
            return
        try:
            self._view.set('status', message)
            if face is not None:
                self._view.set('face', face)
            self._view.update(force=True)
        except Exception as exc:
            logging.debug("[kali] failed to update bootstrap status '%s': %s", message, exc)

    def _reset_kali_session_timer(self):
        self._kali_session_started_at = time.time()
        logging.debug("[kali] session timer reset")

    def kali_session_duration(self):
        started_at = float(self._kali_session_started_at or time.time())
        elapsed = max(int(time.time() - started_at), 0)
        hours = elapsed // 3600
        minutes = (elapsed % 3600) // 60
        seconds = elapsed % 60
        return "%02d:%02d:%02d" % (hours, minutes, seconds)

    def tool_actions(self):
        return list(self._tool_actions)

    def _kali_cfg(self):
        return self._config.get('kali', {}) if isinstance(self._config, dict) else {}

    def _brains_root(self):
        cfg = self._kali_cfg()
        brains_cfg = cfg.get('brains', {}) if isinstance(cfg, dict) else {}
        root = brains_cfg.get('path', DEFAULT_KALI_BRAINS_ROOT) if isinstance(brains_cfg, dict) else DEFAULT_KALI_BRAINS_ROOT
        root = str(root or DEFAULT_KALI_BRAINS_ROOT)
        return root

    def _tool_runtime_context(self):
        return {
            'config': self._config,
            'iface': self._config.get('main', {}).get('iface'),
            'mon_start_cmd': self._config.get('main', {}).get('mon_start_cmd'),
            'ap_ttl': self._config.get('personality', {}).get('ap_ttl'),
            'sta_ttl': self._config.get('personality', {}).get('sta_ttl'),
            'min_rssi': self._config.get('personality', {}).get('min_rssi'),
            'handshakes_path': self._config.get('bettercap', {}).get('handshakes'),
        }

    def _load_last_tool_id(self):
        try:
            if not os.path.isfile(TOOL_STATE_FILE):
                return None
            with open(TOOL_STATE_FILE, 'rt', encoding='utf-8') as fp:
                data = json.load(fp) or {}
            tid = data.get('last_tool')
            return str(tid) if tid else None
        except Exception:
            return None

    def _save_last_tool_id(self, tool_id):
        try:
            data = {
                'last_tool': tool_id,
                'updated_at': int(time.time()),
            }
            with open(TOOL_STATE_FILE, 'wt', encoding='utf-8') as fp:
                json.dump(data, fp)
        except Exception as exc:
            logging.debug("[kali] failed to persist last tool: %s", exc)

    def _refresh_tools(self):
        self._toolbox.reload()
        self._all_tool_actions = self._toolbox.list_actions()

    def _select_boot_tool(self):
        tools = self._toolbox.list_tools()
        if not tools:
            return None

        cfg = self._kali_cfg()
        load_last = bool(cfg.get('load_last_tool', False))
        configured_tool = cfg.get('boot_tool', None)

        # explicit no-tool mode
        if isinstance(configured_tool, str) and configured_tool.lower() in ('none', 'no_tool', 'manual'):
            return None

        if load_last:
            last_tool = self._load_last_tool_id()
            if last_tool in tools:
                return last_tool

        if configured_tool in tools:
            return configured_tool

        return tools[0]

    def _set_active_actions(self):
        if self._active_tool_id is None:
            self._tool_actions = []
            return
        self._tool_actions = [a for a in self._all_tool_actions if a.tool_id == self._active_tool_id]

    def _set_runtime_mode(self, mode):
        self._runtime_mode = str(mode or 'stopped')
        if self._runtime_mode == 'active':
            self.state = self.STATE_RUNNING
        elif self._runtime_mode == 'paused':
            self.state = self.STATE_IDLE
        else:
            self.state = self.STATE_STOPPED
        logging.info("[kali] runtime mode: %s", self._runtime_mode)

    def _sync_env_actions(self):
        if self._model is None or getattr(self._model, 'env', None) is None:
            return
        self._set_bootstrap_status("KALI: syncing tool environment...", faces.SMART)
        try:
            self._model.env.env_method('set_actions', self._tool_actions)
        except Exception:
            try:
                self._model.env.set_actions(self._tool_actions)
            except Exception as exc:
                logging.debug("[kali] failed to refresh env actions for %s: %s", self._active_tool_id, exc)

    def _cache_tool_brain(self, tool_id):
        if not tool_id or tool_id not in self._toolbox.tools:
            return
        self._tool_brains[tool_id] = {
            'model': self._model,
            'reflex': self._reflex,
            'stats': getattr(self, '_stats', None),
            'brain_path': getattr(self, '_nn_path', None)
        }
        logging.debug("[kali] cached brain state for %s", tool_id)

    def _restore_or_load_tool_brain(self, tool_id):
        if tool_id in self._tool_brains:
            self._set_bootstrap_status("KALI: restoring cached brain...", faces.SMART)
            cache = self._tool_brains[tool_id]
            self._model = cache['model']
            self._reflex = cache['reflex']
            self._stats = cache['stats']
            self._nn_path = cache['brain_path']
            logging.info("[kali] restored cached brain for tool %s", tool_id)
            return

        self._set_bootstrap_status("KALI: loading %s brain..." % tool_id, faces.SMART)
        self._configure_tool_brains()
        if self._model is not None:
            self._set_bootstrap_status("KALI: bootstrapping AI...", faces.SMART)
            self._model = ai.load(self._config, self, self._epoch)

    def pause_current_tool(self):
        tool_id = self._active_tool_id
        if not tool_id:
            return {'ok': True, 'error': 'no_active_tool'}
        if self._runtime_mode == 'paused':
            return {'ok': True, 'already_paused': True}

        logging.info("[kali] pausing active tool %s", tool_id)
        self._ai_pause = True
        result = self._toolbox.pause_tool(tool_id, context=self._tool_runtime_context())
        if result.get('ok', False):
            self._set_runtime_mode('paused')
            if self._view:
                self._view.set('status', '%s paused' % tool_id)
        else:
            self._ai_pause = False
        return result

    def resume_current_tool(self):
        tool_id = self._active_tool_id
        if not tool_id:
            return {'ok': False, 'error': 'no_active_tool'}
        if self._runtime_mode != 'paused':
            return {'ok': True, 'already_active': self._runtime_mode == 'active'}

        self._set_bootstrap_status("KALI: resuming tool runtime...", faces.LOOK_R)
        logging.info("[kali] resuming paused tool %s", tool_id)
        result = self._toolbox.resume_tool(tool_id, context=self._tool_runtime_context())
        if result.get('ok', False):
            self._sync_env_actions()
            self._set_runtime_mode('active')
            self._ai_pause = False
            if self._view:
                self._view.set('status', 'KALI: ready')
                self._view.set('face', faces.HAPPY)
                self._view.update(force=True)
        return result

    def stop_current_tool(self, persist=True):
        tool_id = self._active_tool_id
        if not tool_id:
            self._active_tool_id = None
            self._set_active_actions()
            self._sync_env_actions()
            self._set_runtime_mode('stopped')
            self._ai_pause = True
            self._reset_kali_session_timer()
            return {'ok': True}

        logging.info("[kali] stopping active tool %s", tool_id)
        self._ai_pause = True
        self._cache_tool_brain(tool_id)
        result = self._toolbox.deactivate_tool(tool_id, context=self._tool_runtime_context())
        if not result.get('ok', False):
            return result

        self._active_tool_id = None
        self._set_active_actions()
        self._sync_env_actions()
        self._set_runtime_mode('stopped')
        self._reset_kali_session_timer()
        if persist:
            self._save_last_tool_id(None)
        if self._view:
            self._view.set('status', '%s stopped' % tool_id)
        return result

    def _configure_tool_brains(self):
        if not self._active_tool_id:
            logging.info("[kali] skipping brain/reflex reconfiguration because no tool is active")
            return

        self._set_bootstrap_status("KALI: loading %s brain..." % self._active_tool_id, faces.SMART)
        paths = self._toolbox.get_tool_ai_paths(self._active_tool_id, self._brains_root())
        os.makedirs(paths['tool_dir'], exist_ok=True)

        self.configure_brain_path(paths['brain'])

        if self._config['ai'].get('reflex', False):
            self._reflex = ReflexBrain(self._config['main']['iface'], sram_path=paths['reflex'])

        logging.info("[kali] active tool=%s brain=%s reflex=%s", self._active_tool_id, self._nn_path, paths['reflex'])

    def switch_tool(self, tool_id=None, persist=True):
        """
        Switch active tool at runtime without restarting process.
        tool_id=None/'none' performs a full stop of the current tool.
        """
        tools = self._toolbox.list_tools()

        normalized = None if tool_id is None else str(tool_id)
        if normalized and normalized.lower() in ('none', 'no_tool', 'manual'):
            normalized = None

        if normalized is None:
            return self.stop_current_tool(persist=persist)

        if normalized is not None and normalized not in tools:
            raise ValueError("unknown tool: %s" % normalized)

        prev = self._active_tool_id

        if prev == normalized:
            logging.info("[kali] tool %s is already active, switch ignored", prev)
            return {'ok': True, 'already_active': True}

        logging.info("[kali] switching tool %s -> %s", prev, normalized)
        self._ai_pause = True
        self._cache_tool_brain(prev)
        if prev:
            down = self._toolbox.deactivate_tool(prev, context=self._tool_runtime_context())
            if not down.get('ok', False):
                return down

        self._active_tool_id = normalized
        self._set_active_actions()
        self._restore_or_load_tool_brain(normalized)
        self._sync_env_actions()

        self._set_bootstrap_status("KALI: resuming tool runtime...", faces.LOOK_R)
        up = self._toolbox.activate_tool(normalized, context=self._tool_runtime_context())
        if not up.get('ok', False):
            self._active_tool_id = None
            self._set_active_actions()
            self._sync_env_actions()
            self._set_runtime_mode('stopped')
            return up

        if persist:
            self._save_last_tool_id(self._active_tool_id)

        if self._started and bool(self._config.get('ai', {}).get('enabled', False)) and self._active_tool_id and self._model is None:
            logging.info("[kali] tool activated while ai.enabled=true -> starting RL loop")
            self.start_ai()

        self._set_runtime_mode('active')
        self._ai_pause = False
        if self._view:
            self._view.set('status', 'KALI: ready')
            self._view.set('face', faces.HAPPY)
            self._view.update(force=True)
        logging.info("[kali] switched tool: %s -> %s", prev, self._active_tool_id)
        plugins.on('kali_tool_switched', self, prev, self._active_tool_id)
        return up

    def execute_tool_action(self, action_idx: int, context=None):
        if self._runtime_mode != 'active':
            return {'ok': False, 'error': 'tool_not_active', 'signals': {}, 'metrics': {}, 'costs': {}}

        if self._active_tool_id is None:
            return {'ok': False, 'error': 'no_tool_mode', 'signals': {}, 'metrics': {}, 'costs': {}}

        if not self._tool_actions:
            return {'ok': False, 'error': 'no_actions', 'signals': {}, 'metrics': {}, 'costs': {}}

        action_idx = max(0, min(int(action_idx), len(self._tool_actions) - 1))
        action = self._tool_actions[action_idx]
        result = self._toolbox.execute(action, context=context or {})

        if 'ui' in result and isinstance(result['ui'], dict):
            ui_data = result['ui']
            if 'face' in ui_data:
                self._view.set('face', ui_data['face'])
            if 'status' in ui_data:
                self._view.set('status', str(ui_data['status']))

        plugins.on('kali_tool_action', self, action.tool_id, action.command_id, result)
        return result

    def observe_toolbox(self):
        return self._toolbox.observe()

    def run(self, command, verbose_errors=True):
        # compatibility bridge: route to currently active tool adapter if available
        adapter = self._toolbox.adapters.get(self._active_tool_id) if self._active_tool_id else None
        if adapter is not None and hasattr(adapter, 'client') and hasattr(adapter.client, 'run'):
            return adapter.client.run(command, verbose_errors=verbose_errors)

        logging.warning("[kali] run('%s') requested but no compatible command client is active", command)
        return None

    def _active_tool_client(self):
        adapter = self._toolbox.adapters.get(self._active_tool_id) if self._active_tool_id else None
        client = getattr(adapter, 'client', None) if adapter is not None else None
        return client

    async def _on_tool_event(self, msg):
        try:
            jmsg = json.loads(msg)
        except Exception:
            return

        tag = str(jmsg.get('tag', '') or '')
        data = jmsg.get('data', {}) if isinstance(jmsg.get('data', {}), dict) else {}
        if tag:
            try:
                plugins.on('bcap_%s' % re.sub(r"[^a-z0-9_]+", "_", tag.lower()), self, jmsg)
            except Exception:
                pass

        if tag == 'sys.log':
            msg_txt = str(data.get('message') or jmsg.get('message') or '')
            low = msg_txt.lower()
            if 'wifi could not inject wifi packet' in low and 'resource temporarily unavailable' in low:
                self._epoch.track_interaction(success=False, error=True)
                now = time.time()
                if now - self._last_injection_warn >= 5.0:
                    self._last_injection_warn = now
                    logging.warning("[kali] injection pressure detected: %s", msg_txt)

    def _event_poller(self):
        while not self._event_stop.is_set():
            client = self._active_tool_client()
            if client is None:
                time.sleep(2)
                continue

            try:
                client.run('events.clear', verbose_errors=False)
            except Exception:
                pass

            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                loop.run_until_complete(client.start_websocket(self._on_tool_event))
            except Exception as exc:
                logging.warning("[kali] websocket poller reconnecting after error: %s", exc)
                time.sleep(2)
            finally:
                try:
                    loop.close()
                except Exception:
                    pass

    def _start_event_polling(self):
        if self._event_thread and self._event_thread.is_alive():
            return
        self._event_stop.clear()
        self._event_thread = threading.Thread(target=self._event_poller, name='kali-event-poller', daemon=True)
        self._event_thread.start()

    def start(self):
        if self._started:
            logging.debug("[kali] start() ignored; already started")
            return

        self._started = True
        self._reset_kali_session_timer()
        self._auto_stop.clear()

        try:
            self.set_starting()
            self._set_bootstrap_status("KALI: bootstrapping AI...", faces.SMART)
            self._refresh_tools()

            boot_tool = self._select_boot_tool()
            self._active_tool_id = boot_tool
            self._set_active_actions()

            if self._active_tool_id:
                self._configure_tool_brains()
                self._set_bootstrap_status("KALI: resuming tool runtime...", faces.LOOK_R)
                up = self._toolbox.activate_tool(self._active_tool_id, context=self._tool_runtime_context())
                if not up.get('ok', False):
                    raise RuntimeError("failed to load boot tool %s: %s" % (self._active_tool_id, up.get('error', 'unknown')))
                self._set_runtime_mode('active')
            else:
                self._toolbox.deactivate_tool(context=self._tool_runtime_context())
                self._set_runtime_mode('stopped')

            self._save_last_tool_id(self._active_tool_id)
            logging.info("[kali] boot tool=%s (actions=%d)", self._active_tool_id, len(self._tool_actions))
            self._start_event_polling()

            if self._config['personality'].get('advertise', False):
                self.start_advertising()

            ai_enabled = bool(self._config.get('ai', {}).get('enabled', False))
            tool_active = bool(self._active_tool_id and len(self._tool_actions) > 0)

            if ai_enabled and tool_active:
                self._set_bootstrap_status("KALI: bootstrapping AI...", faces.SMART)
                logging.info("[kali] ai.enabled=true and tool active -> starting RL loop")
                self.start_ai()
            else:
                if ai_enabled and not tool_active:
                    logging.info("[kali] ai.enabled=true but no active tool -> autonomous idle until tool is enabled")
                else:
                    logging.info("[kali] ai.enabled=false -> starting autonomous loop")
                self._start_auto_mode()

            self._set_bootstrap_status("KALI: ready", faces.HAPPY)
            self.set_ready()
        except Exception as exc:
            logging.exception("[kali] start failed: %s", exc)

    def stop(self):
        self._auto_stop.set()
        self._event_stop.set()
        self._ai_pause = True
        self._reset_kali_session_timer()

        client = self._active_tool_client()
        if client and hasattr(client, 'running'):
            client.running = False

        try:
            if self._active_tool_id:
                self._cache_tool_brain(self._active_tool_id)
                self._toolbox.deactivate_tool(self._active_tool_id, context=self._tool_runtime_context())
        except Exception:
            pass
        self._active_tool_id = None
        self._set_active_actions()
        self._set_runtime_mode('stopped')

    def _start_auto_mode(self):
        if self._auto_thread and self._auto_thread.is_alive():
            return

        self._auto_thread = threading.Thread(target=self._auto_loop, name='kali-auto-loop', daemon=True)
        self._auto_thread.start()

    def _auto_loop(self):
        while not self._auto_stop.is_set():
            try:
                # no-tool mode = manual-equivalent idle mode
                if self._runtime_mode != 'active' or self._active_tool_id is None:
                    time.sleep(2)
                    continue

                if not self._tool_actions:
                    self._refresh_tools()
                    self._set_active_actions()
                    time.sleep(2)
                    continue

                action_idx = self._select_auto_action()
                context = {
                    'epoch': self._epoch.epoch,
                    'state': self._epoch.data() if self._epoch.data() else self._epoch.get_state(),
                }
                result = self.execute_tool_action(action_idx, context=context)

                snapshot = self.observe_toolbox()
                self._epoch.observe(snapshot)
                self._epoch.ingest_execution(result)
                self.next_epoch()

                self._view.set('uptime', time.strftime('%H:%M:%S', time.gmtime(pwnagotchi.uptime())))

                loop_delay = float(self._config.get('personality', {}).get('recon_time', 5) or 5)
                time.sleep(max(loop_delay, 1.0))
            except Exception as exc:
                logging.exception("[kali] auto loop error: %s", exc)
                time.sleep(2)

    def _select_auto_action(self) -> int:
        preferred = [
            'explore_environment',
            'focus_context',
            'interact_with_target',
            'force_state_change',
            'restart_recon',
        ]

        by_name = {a.command_id: idx for idx, a in enumerate(self._tool_actions)}
        for name in preferred:
            if name in by_name:
                idx = by_name[name]
                self._auto_action_ptr = (self._auto_action_ptr + 1) % max(len(self._tool_actions), 1)
                return idx

        idx = self._auto_action_ptr % len(self._tool_actions)
        self._auto_action_ptr = (self._auto_action_ptr + 1) % len(self._tool_actions)
        return idx

    def _save_recovery_data(self):
        try:
            logging.warning("writing recovery data to %s ...", RECOVERY_DATA_FILE)
            with open(RECOVERY_DATA_FILE, 'w') as fp:
                data = {
                    'started_at': self._started_at,
                    'epoch': self._epoch.epoch,
                    'last_state': self._epoch.data(),
                    'active_tool': self._active_tool_id,
                }
                json.dump(data, fp)
        except Exception as exc:
            logging.error("[kali] failed to save recovery data: %s", exc)

    def _restart(self):
        self.set_rebooting()
        pwnagotchi.reboot()
