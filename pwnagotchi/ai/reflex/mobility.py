# pwnagotchi/ai/reflex/mobility.py

import time
import math


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class MobilityContext:
    """
    Reflex-level mobility estimator.

    Converts raw GPS updates into a stable mobility_pressure scalar [0.0 - 1.0].

    This module does NOT modify behavior directly.
    It only exposes contextual pressure to be consumed by ReflexBrain.
    """

    def __init__(
        self,
        max_expected_speed=15.0,      # m/s (~54 km/h)
        burst_max=5.0,                # m/s delta considered strong burst
        smooth_alpha=0.15,            # low-pass filter for velocity
        pressure_alpha=0.1,           # smoothing for final pressure
        stationary_threshold=0.3,     # m/s under which considered static
        stationary_hold_time=10.0     # seconds below threshold to lock stationary
    ):
        self.max_expected_speed = max_expected_speed
        self.burst_max = burst_max
        self.smooth_alpha = smooth_alpha
        self.pressure_alpha = pressure_alpha
        self.stationary_threshold = stationary_threshold
        self.stationary_hold_time = stationary_hold_time

        self._last_gps = None
        self._last_ts = None

        self.instant_speed = 0.0
        self.smooth_speed = 0.0
        self.burst = 0.0
        self.bearing = 0.0

        self._mobility_pressure = 0.0

        self._below_threshold_since = None
        self._is_stationary = True

    # ---------------------------------------------------------

    def update(self, lat, lon):
        """
        Update context with new GPS coordinate.
        Returns mobility_pressure ∈ [0.0, 1.0]
        """
        now = time.time()

        if self._last_gps is None:
            self._last_gps = (lat, lon)
            self._last_ts = now
            return self._mobility_pressure

        dt = now - self._last_ts
        if dt <= 0:
            return self._mobility_pressure

        # --- Haversine distance ---
        R = 6371000.0  # meters

        lat1 = math.radians(self._last_gps[0])
        lon1 = math.radians(self._last_gps[1])
        lat2 = math.radians(lat)
        lon2 = math.radians(lon)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        dist = R * c

        self.instant_speed = dist / dt

        # --- Bearing calculation ---
        y = math.sin(dlon) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        self.bearing = (math.degrees(math.atan2(y, x)) + 360) % 360

        # --- Smooth speed (low-pass filter) ---
        self.smooth_speed = (
            self.smooth_alpha * self.instant_speed +
            (1.0 - self.smooth_alpha) * self.smooth_speed
        )

        # --- Burst detection ---
        self.burst = max(0.0, self.instant_speed - self.smooth_speed)

        # --- Stationary hysteresis ---
        if self.smooth_speed < self.stationary_threshold:
            if self._below_threshold_since is None:
                self._below_threshold_since = now
            elif now - self._below_threshold_since >= self.stationary_hold_time:
                self._is_stationary = True
        else:
            self._below_threshold_since = None
            self._is_stationary = False

        # --- Normalize components ---
        speed_norm = clamp(self.smooth_speed / self.max_expected_speed, 0.0, 1.0)
        burst_norm = clamp(self.burst / self.burst_max, 0.0, 1.0)

        raw_pressure = 0.7 * speed_norm + 0.3 * burst_norm

        # If locked stationary, suppress noise completely
        if self._is_stationary:
            raw_pressure = 0.0

        # --- Smooth final mobility pressure ---
        self._mobility_pressure = (
            self.pressure_alpha * raw_pressure +
            (1.0 - self.pressure_alpha) * self._mobility_pressure
        )

        # Update state
        self._last_gps = (lat, lon)
        self._last_ts = now

        return self._mobility_pressure

    # ---------------------------------------------------------

    @property
    def mobility_pressure(self):
        return self._mobility_pressure

    @property
    def is_stationary(self):
        return self._is_stationary
