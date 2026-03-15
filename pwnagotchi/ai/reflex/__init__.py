# pwnagotchi/ai/reflex/__init__.py
import logging
import numpy as np
import re
import subprocess
import psutil
import time
import json
import os

from .mobility import MobilityContext
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
        # ACTION: Discrete choices for immediate reaction (e.g., 0: Stay, 1: Hop, 2: Deauth)
        self.action_space = spaces.Discrete(3)

        # OBSERVATION: Key "Oikos" metrics like RSSI, CPU Temp, and blindbug symptoms
        self.observation_space = spaces.Dict({
            "rssi": spaces.Box(low=-100, high=0, shape=(1,), dtype=np.float32),
            "temp": spaces.Box(low=30, high=90, shape=(1,), dtype=np.float32),
            "handshake_delta": spaces.Box(low=0, high=10, shape=(1,), dtype=np.float32),
            "timeout_errors": spaces.Discrete(10),
            "injection_errors": spaces.Discrete(10),
            "io_wait": spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
            "is_promiscuous": spaces.Discrete(2),
            "bc_errors": spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32)
        })

        self.state = {}
        self.last_handshakes = 0

    def _get_obs(self):
        # Clamp RSSI to prevent ghost APs
        rssi = np.clip(float(self.state.get('rssi', -100.0)), -90, -45)
        temp = float(self.state.get('temperature', 40.0))

        current_handshakes = float(self.state.get('num_handshakes', 0))
        handshake_delta = max(0.0, current_handshakes - self.last_handshakes)

        timeout_errors = min(int(self.state.get('timeout_errors', 0)), 9)
        injection_errors = min(int(self.state.get('injection_errors', 0)), 9)
        io_wait = float(self.state.get('io_wait', 0.0))
        is_promiscuous = int(self.state.get('is_promiscuous', 0))
        bc_errors = min(float(self.state.get('num_bc_errors', 0)) / 10.0, 1.0)

        return {
            "rssi": np.array([rssi], dtype=np.float32),
            "temp": np.array([temp], dtype=np.float32),
            "handshake_delta": np.array([handshake_delta], dtype=np.float32),
            "timeout_errors": timeout_errors,
            "injection_errors": injection_errors,
            "io_wait": np.array([io_wait], dtype=np.float32),
            "is_promiscuous": is_promiscuous,
            "bc_errors": np.array([bc_errors], dtype=np.float32)
        }

    def _calculate_reflex_reward(self):
        # Positive for handshakes, negative for 'blindbug' symptoms
        reward = 0.0

        # Reward for handshakes
        current_handshakes = float(self.state.get('num_handshakes', 0))
        delta = max(0.0, current_handshakes - self.last_handshakes)
        if delta > 0:
            reward += 5.0 * delta

        # Penalty for blindness (blindbug symptom)
        if self.state.get('blind_for_epochs', 0) > 0 and delta == 0:
            reward -= 0.05 * self.state.get('blind_for_epochs', 0)

        # --- Handshake starvation penalty ---
        assoc_count = float(self.state.get('num_associations', 0))
        if assoc_count > 10 and delta == 0:
            reward -= 0.1 * (assoc_count - 10)  # slight frustration

        return reward

    def _detect_blindbug(self):
        # Terminal state if blind for too long (e.g. > 10 epochs)
        return self.state.get('blind_for_epochs', 0) > 10

    def _detect_systemic_collapse(self):
        """
        Detects if the system has entered a 'Brain Dead' state
        where communication with the environment is severed.
        """
        # If we have reached a high number of timeout errors or consecutive
        # blind epochs, the 'Oikos' is effectively dead to the agent.
        is_collapsed = (
            int(self.state.get('timeout_errors', 0)) > 8 or
            int(self.state.get('blind_for_epochs', 0)) > 40
        )
        return is_collapsed

    def step(self, action):
        # Process the action (tactical intervention)
        # ... your logic here ...

        # Check for systemic collapse
        collapsed = self._detect_systemic_collapse()

        if collapsed:
            logging.warning("[Reflex] SYSTEMIC COLLAPSE DETECTED. Initiating Reincarnation.")
            # Trigger a 'reincarnation' (reboot) via subprocess
            # This is the 'death/rebirth' cycle of the agent
            subprocess.Popen(['sudo', 'reboot'])

        reward = self._calculate_reflex_reward()
        terminated = self._detect_blindbug()
        truncated = False

        # Update history for next step
        self.last_handshakes = float(self.state.get('num_handshakes', 0))

        obs = self._get_obs()
        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.last_handshakes = float(self.state.get('num_handshakes', 0))
        return self._get_obs(), {}


class ReflexBrain:
    """
    Fast, volatile, disposable.
    No SB3 imports.
    No config access.
    """
    def __init__(self, iface='mon0'):
        self.env = ReflexEnv()
        # CONTEXT: Mobility pressure from GPS (0.0 to 1.0)
        self.mobility = MobilityContext()
        self.iface = iface
        self.state = {}
        self.familiarity = {}  # MAC -> score (0.0 to 1.0)
        self.patterns = {
            'timeout': re.compile(r'brcmf_.*failed.*-110'),
            'injection_fail': re.compile(r'wifi could not inject WiFi packet.*Resource temporarily unavailable'),
            'promisc_enter': re.compile(r'%s.*: entered promiscuous mode' % iface),
            'promisc_left': re.compile(r'%s.*: left promiscuous mode' % iface),
            'association': re.compile(r'sending association frame to .* \(([0-9a-fA-F:]{17})'),
            'handshake': re.compile(r'captured handshake from .*?([0-9a-fA-F:]{17})')
        }
        self.last_handshakes = 0
        self.frustration = 0.0
        
        self.ticker_period = 10.0        # seconds (starting point)
        self.ticker_min = 1.0 # 5.0
        self.ticker_max = 30.0

        self._inj_fail_streak = 0
        self._success_streak = 0
        self._last_adjust_ts = 0.0
        self._cmd_latency = 0.0
        self._inj_rate = 0.0
        self._log_window = 1.0
        self._last_log_check = time.time()
        self._is_promiscuous = 0
        self._cached_log_metrics = {'timeout_errors': 0, 'injection_errors': 0, 'io_wait': 0.0, 'is_promiscuous': 0}
        self.timeout_errors = 0
        self.injection_errors = 0
        
        self.sram = ReflexSRAM()
        self._baseline = {
            "safe_interaction_scale": 1.0,
            "safe_deauth_cooldown": 0.0,
            "safe_assoc_cooldown": 0.0,
            "last_stable_pressure": 0.0
        }
        self._last_save = 0
        self._current_deauth_cooldown = 0.0
        self._current_assoc_cooldown = 0.0
        self._current_interaction_scale = 1.0
        
        self._load_state()

        # Neonatal phase init
        self._phase = "NEONATAL"
        self._boot_ts = time.time()
        self._phase_ts = time.time()
        self._stable_ticks = 0

    def _load_state(self):
        data = self.sram.load()
        
        # Migration: Check legacy JSON if SRAM is empty
        if not data and os.path.exists("/root/.reflex_state.json"):
            try:
                with open("/root/.reflex_state.json", 'r') as f:
                    data = json.load(f)
                logging.info("[Reflex] Migrated legacy state to SRAM.")
            except Exception:
                pass

        if data:
            # Decay across cold boots (return to default)
            decay = 0.95
            
            # Cooldowns: decay towards 0.0 (less delay)
            self._baseline["safe_deauth_cooldown"] = data.get("safe_deauth_cooldown", 0.0) * decay
            self._baseline["safe_assoc_cooldown"] = data.get("safe_assoc_cooldown", 0.0) * decay
            
            # Scale: decay towards 1.0 (more aggressive/less restriction)
            saved_scale = data.get("safe_interaction_scale", 1.0)
            self._baseline["safe_interaction_scale"] = 1.0 - (1.0 - saved_scale) * decay
            
            self._baseline["last_stable_pressure"] = data.get("last_stable_pressure", 0.0)
            
            logging.info(f"[Reflex] Loaded baseline state: {self._baseline}")

    def _maybe_persist(self):
        now = time.time()
        if now - self._last_save < 300:
            return

        # Save only when calm and stable
        if self.stress_level < 0.2 and self._inj_fail_streak == 0:
            # Monotonic tightening: remember the safe limits found during operation
            self._baseline["safe_interaction_scale"] = min(self._baseline["safe_interaction_scale"], self._current_interaction_scale)
            self._baseline["safe_deauth_cooldown"] = max(self._baseline["safe_deauth_cooldown"], self._current_deauth_cooldown)
            self._baseline["safe_assoc_cooldown"] = max(self._baseline["safe_assoc_cooldown"], self._current_assoc_cooldown)
            self._baseline["last_stable_pressure"] = self.stress_level
            
            data = {
                "safe_interaction_scale": self._baseline["safe_interaction_scale"],
                "safe_deauth_cooldown": self._baseline["safe_deauth_cooldown"],
                "safe_assoc_cooldown": self._baseline["safe_assoc_cooldown"],
                "last_stable_pressure": self._baseline["last_stable_pressure"],
                "hardware_signature": f"{self.iface}",
                "last_update": int(now)
            }
            
            self.sram.save(data)
            self._last_save = now

    def _body_ready(self):
        return (
            self._inj_fail_streak == 0 and
            self._cmd_latency < 0.5 and
            self.state.get("timeout_errors", 0) == 0 and
            self.state.get("io_wait", 0.0) < 20.0 and
            self.state.get("is_promiscuous", 0) in (0, 1)
        )

    def update_latency(self, dt):
        # Smooth the latency to avoid jitter, but capture spikes
        # If latency is high, it means the driver/bettercap is struggling
        alpha = 0.2
        self._cmd_latency = (1 - alpha) * self._cmd_latency + alpha * dt

    def _check_logs(self):
        now = time.time()
        if now - self._last_log_check < 2.0:
            return self._cached_log_metrics

        self._log_window = max(1.0, now - self._last_log_check)

        metrics = {'timeout_errors': 0, 'injection_errors': 0, 'io_wait': 0.0, 'is_promiscuous': self._is_promiscuous}
        try:
            metrics['io_wait'] = psutil.cpu_times_percent(interval=None).iowait
        except Exception:
            pass

        try:
            # Check kernel logs for timeouts
            cmd = ['journalctl', '-k', '--since', f'@{self._last_log_check}', '--no-pager']
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, _ = proc.communicate()
            klogs = out.decode('utf-8', errors='ignore')
            metrics['timeout_errors'] = len(self.patterns['timeout'].findall(klogs))

            # Check general logs for promiscuous mode and injection errors
            cmd = ['journalctl', '--since', f'@{self._last_log_check}', '--no-pager']
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, _ = proc.communicate()
            logs = out.decode('utf-8', errors='ignore')

            metrics['injection_errors'] = len(self.patterns['injection_fail'].findall(logs))

            # Update familiarity from logs (Taste)
            for mac in self.patterns['association'].findall(logs):
                self.familiarity[mac] = 1.0
            for mac in self.patterns['handshake'].findall(logs):
                self.familiarity[mac] = 1.0

            enters = [m.start() for m in self.patterns['promisc_enter'].finditer(logs)]
            lefts = [m.start() for m in self.patterns['promisc_left'].finditer(logs)]

            if enters and lefts:
                self._is_promiscuous = 1 if max(enters) > max(lefts) else 0
            elif enters:
                self._is_promiscuous = 1
            elif lefts:
                self._is_promiscuous = 0
            
            metrics['is_promiscuous'] = self._is_promiscuous
        except Exception as e:
            logging.debug("[ReflexBrain] Log check error: %s" % e)
        
        self._last_log_check = now
        self._cached_log_metrics = metrics
        return metrics

    def regulate_ticker(self, inj_rate=0.0):
        # Sane ticker behavior:
        # 1. Fast ramp up on pain (approach target)
        # 2. Unconditional slow decay (always drift to baseline)
        
        threshold = 0.5 # errors/sec considered "full saturation"
        error_pressure = min(inj_rate / threshold, 1.0)
        
        # Target is where the ticker WANTS to be based on current pain
        target = self.ticker_min + error_pressure * (self.ticker_max - self.ticker_min)
        
        old_ticker = self.ticker_period

        if target > self.ticker_period:
            # Fast ramp up (0.4)
            self.ticker_period += (target - self.ticker_period) * 0.4
        else:
            # Slow decay (0.05)
            self.ticker_period += (target - self.ticker_period) * 0.05
            
        self.ticker_period = max(self.ticker_period, self.ticker_min)

        if abs(self.ticker_period - old_ticker) > 0.5:
             logging.warning(f"[Reflex] Ticker adjusted: {old_ticker:.2f}s -> {self.ticker_period:.2f}s (pressure: {error_pressure:.2f})")

    def observe(self, observation):
        """
        Receives the SAME observation as the gym env.
        """
        if observation is not self.state:
            self.timeout_errors = 0
            self.injection_errors = 0

        gps = observation.get('gps')
        mobility_pressure = 0.0
        if gps and isinstance(gps, dict):
            lat = gps.get('Latitude') or gps.get('lat')
            lon = gps.get('Longitude') or gps.get('lon')
            if lat is not None and lon is not None:
                mobility_pressure = self.mobility.update(lat, lon)
            else:
                mobility_pressure = self.mobility.mobility_pressure
        else:
            mobility_pressure = self.mobility.mobility_pressure

        observation['mobility_pressure'] = mobility_pressure

        log_metrics = self._check_logs()
        
        self.timeout_errors += log_metrics.get('timeout_errors', 0)
        self.injection_errors += log_metrics.get('injection_errors', 0)
        
        observation['timeout_errors'] = self.timeout_errors
        observation['injection_errors'] = self.injection_errors
        observation['io_wait'] = log_metrics.get('io_wait', 0)
        observation['is_promiscuous'] = log_metrics.get('is_promiscuous', 0)
        observation['num_bc_errors'] = log_metrics.get('num_bc_errors', 0)

        # Update streaks
        inj_errs = log_metrics.get('injection_errors', 0)
        if inj_errs > 0:
            self._inj_fail_streak += 1
            self._success_streak = 0
        else:
            self._success_streak += 1
            self._inj_fail_streak = 0

        # --- Frustration / boredom loop ---
        assoc_count = float(observation.get('num_associations', 0))
        current_handshakes = float(observation.get('num_handshakes', 0))
        self.frustration = max(0.0, assoc_count - current_handshakes) / 10.0

        # Decay familiarity (Habituation)
        dead_keys = []
        for mac in self.familiarity:
            decay_factor = 0.85 if assoc_count > 0 and current_handshakes == 0 else 0.9
            self.familiarity[mac] *= decay_factor
            if self.familiarity[mac] < 0.1:
                dead_keys.append(mac)
        for k in dead_keys:
            del self.familiarity[k]

        # Regulate ticker
        inj_rate = inj_errs / self._log_window
        self._inj_rate = inj_rate
        self.regulate_ticker(inj_rate)

        # TRIGGER: Endorphin as last resort
        if (
            self._inj_fail_streak > 10 and
            self.ticker_period >= self.ticker_max
        ):
            #self.endorphin()
            # Reset the error count in the current state so we don't loop the reset
            observation['injection_errors'] = 0
            self._inj_fail_streak = 0

        self.state = observation
        self.env.state = observation

        # --- Neonatal / calibration phase logic ---
        now = time.time()
        phase_duration = now - self._phase_ts

        if self._phase == "NEONATAL":
            if self._body_ready():
                self._stable_ticks += 1
            else:
                self._stable_ticks = 0

            # Require several calm observations in a row OR timeout (2m)
            if self._stable_ticks >= 5 or phase_duration > 120:
                if phase_duration > 120:
                    logging.warning("[Reflex] NEONATAL phase timed out. Forcing CALIBRATING.")
                else:
                    logging.info("[Reflex] Body stabilized -> CALIBRATING")
                self._phase = "CALIBRATING"
                self._phase_ts = now
                self._stable_ticks = 0

        elif self._phase == "CALIBRATING":
            if self._body_ready():
                self._stable_ticks += 1
            else:
                self._stable_ticks = 0

            if self._stable_ticks >= 10 or phase_duration > 120:
                if phase_duration > 120:
                    logging.warning("[Reflex] CALIBRATING phase timed out. Forcing OPERATIONAL.")
                else:
                    logging.info("[Reflex] Calibration complete -> OPERATIONAL")
                self._phase = "OPERATIONAL"
                self._phase_ts = now

    def bias(self):
        """
        Returns soft modifiers ONLY.
        Returns soft modifiers based on the 'Oikos' (habitat) state.
        """
        modifiers = {
            "recon_time_multiplier": 1.0,
            "min_rssi_offset": 0,
            "interaction_scale": 1.0,
            "ticker_period": self.ticker_period,
            "deauth_cooldown": 0.0,
            "assoc_cooldown": 0.0,
            "ttl_multiplier": 1.0
        }

        mobility = self.state.get('mobility_pressure', 0.0)
        modifiers["ttl_multiplier"] = 1.0 - (0.6 * mobility)

        if self._phase == "NEONATAL":
            return {
                "recon_time_multiplier": 2.5,
                "min_rssi_offset": 0,
                "interaction_scale": 0.15,
                "ticker_period": max(self.ticker_period, self.ticker_max),
                "deauth_cooldown": 5.0,
                "assoc_cooldown": 2.0
            }

        if self._phase == "CALIBRATING":
            return {
                "recon_time_multiplier": 1.5,
                "min_rssi_offset": -5,
                "interaction_scale": 0.4,
                "ticker_period": self.ticker_period,
                "deauth_cooldown": 2.0,
                "assoc_cooldown": 1.0
            }

        stress = self.stress_level  # 0.0 to 1.0
        risk = self.risk            # 0.0 to 1.0
        boredom = self.boredom      # 0.0 to 1.0

        # Calculate delta for Flow State
        current_handshakes = float(self.state.get('num_handshakes', 0))
        delta = max(0.0, current_handshakes - self.last_handshakes)
        self.last_handshakes = current_handshakes

        # MORIN PHILOSOPHY: Self-Preservation vs. Evolutionary Leap
        if (stress > 0.8 and self.ticker_period >= self.ticker_max) or risk > 0.8:
            # THE "FINAL PUSH": If death is imminent,
            # prioritize wide scanning over heavy interaction to save the driver.
            logging.info("[Reflex] CRITICAL STATE: Maximizing scan coverage before crash.")
            modifiers["recon_time_multiplier"] = 0.5  # Move faster
            modifiers["interaction_scale"] = 0.2     # Suppress injection to save driver
            modifiers["min_rssi_offset"] = -15       # Listen to everything

        elif stress > 0.5:
            # CAUTIOUS PRESERVATION: Slow down to cool the hardware.
            modifiers["recon_time_multiplier"] = 2.0
            modifiers["interaction_scale"] = 0.5

        # Flow State (High Performance)
        if stress < 0.3 and delta > 0:
            # We are in the "Zone": stable hardware + successful captures
            # Let's push the boundary.
            modifiers["interaction_scale"] = 1.5
            modifiers["recon_time_multiplier"] = 0.8  # Move faster

        # Resource Unavailable Detection
        if self._inj_fail_streak > 0:
            modifiers["interaction_scale"] *= max(
                0.0, 1.0 - (0.3 * self._inj_fail_streak)
            )

        # HABITUATION: If bored, explore more (Taste)
        if boredom > 0.3:
            # Scan longer to find new flavors
            modifiers["recon_time_multiplier"] = max(modifiers["recon_time_multiplier"], 1.0 + boredom)
            # Listen to fainter signals (expand the menu)
            modifiers["min_rssi_offset"] = -10 * boredom

        # --- Never fully block deauth ---
        min_interaction_scale = 0.2
        modifiers["interaction_scale"] = max(modifiers["interaction_scale"], min_interaction_scale)

        # --- Frustration pushes aggression ---
        modifiers["interaction_scale"] *= 1.0 + 0.5 * self.frustration
        modifiers["recon_time_multiplier"] *= 1.0 + 0.2 * self.frustration

        # Calculate pressure for cooldowns (Local Pacing)
        # Pressure = Injection Errors + Latency
        pressure = min(1.0, (self._inj_rate * 2.0) + (self._cmd_latency * 2.0))
        
        if pressure > 0.1:
            modifiers["deauth_cooldown"] = pressure * 2.5
            modifiers["assoc_cooldown"] = pressure * 1.0
            
        # Apply baseline floor (safety ratchet)
        modifiers["deauth_cooldown"] = max(modifiers["deauth_cooldown"], self._baseline["safe_deauth_cooldown"])
        modifiers["assoc_cooldown"] = max(modifiers["assoc_cooldown"], self._baseline["safe_assoc_cooldown"])
        
        # Apply baseline cap for interaction scale (safety ratchet)
        modifiers["interaction_scale"] = min(modifiers["interaction_scale"], self._baseline["safe_interaction_scale"])

        # Panic Mode: High Bettercap error rate
        num_bc_errors = float(self.state.get('num_bc_errors', 0))
        if num_bc_errors > 5:
             logging.warning("[Reflex] PANIC: High Bettercap error rate (%.1f). Suppressing interaction.", num_bc_errors)
             modifiers["interaction_scale"] = 0.01

        self._current_deauth_cooldown = modifiers["deauth_cooldown"]
        self._current_assoc_cooldown = modifiers["assoc_cooldown"]
        self._current_interaction_scale = modifiers["interaction_scale"]
        
        self._maybe_persist()

        return modifiers

    @property
    def stress_level(self):
        # 0.0 to 1.0 based on io_wait, injection_errors, temp
        io = float(self.state.get('io_wait', 0)) / 100.0
        # Injection errors are critical. Scale so ~50 errors = 100% stress.
        inj = min(float(self.state.get('injection_errors', 0)), 50.0) / 50.0
        temp = max(0.0, float(self.state.get('temperature', 40)) - 50.0) / 30.0
        # Latency > 1.0s is considered critical stress
        lat = min(self._cmd_latency, 1.0)
        bc_err = min(float(self.state.get('num_bc_errors', 0)) / 10.0, 1.0)
        return min(1.0, max(io, inj, temp, lat, bc_err))

    @property
    def risk(self):
        # 0.0 to 1.0 based on timeout_errors, blind_for_epochs
        # Timeouts are less dramatic. Allow up to 15 before maxing risk.
        to = min(float(self.state.get('timeout_errors', 0)), 20.0) / 100.0
        blind = min(float(self.state.get('blind_for_epochs', 0)), 10.0) / 10.0
        return min(1.0, max(to, blind))

    @property
    def boredom(self):
        if not self.familiarity:
            return 0.0
        return sum(self.familiarity.values()) / len(self.familiarity)
