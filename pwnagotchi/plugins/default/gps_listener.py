import json
import logging
import os
import subprocess
import threading

import pwnagotchi.plugins as plugins
import pwnagotchi.ui.fonts as fonts
from pwnagotchi.ui.components import LabeledValue
from pwnagotchi.ui.view import BLACK

"""
# Android
# Termux:API : https://f-droid.org/en/packages/com.termux.api/
# Termux : https://f-droid.org/en/packages/com.termux/
pkg install termux-api socat bc

-----
#!/data/data/com.termux/files/usr/bin/bash

# Server details
SERVER_IP="192.168.44.44"  # IP of the socat receiver
SERVER_PORT="5000"         # UDP port to send data to

# Function to calculate checksum
calculate_checksum() {
  local sentence="$1"
  local checksum=0
  # Loop through each character in the sentence
  for ((i = 0; i < ${#sentence}; i++)); do
    checksum=$((checksum ^ $(printf '%d' "'${sentence:i:1}")))
  done
  # Return checksum in hexadecimal
  printf "%02X" $checksum
}

# Infinite loop to send GPS data
while true; do
  # Get location data
  LOCATION=$(termux-location -p gps)

  # Extract latitude, longitude, altitude, speed, and bearing
  LATITUDE=$(echo "$LOCATION" | jq '.latitude')
  LONGITUDE=$(echo "$LOCATION" | jq '.longitude')
  ALTITUDE=$(echo "$LOCATION" | jq '.altitude')
  SPEED=$(echo "$LOCATION" | jq '.speed') # Speed in meters per second
  BEARING=$(echo "$LOCATION" | jq '.bearing')

  # Convert speed from meters per second to knots and km/h
  SPEED_KNOTS=$(echo "$SPEED" | awk '{printf "%.1f", $1 * 1.943844}')
  SPEED_KMH=$(echo "$SPEED" | awk '{printf "%.1f", $1 * 3.6}')

  # Format latitude and longitude for NMEA
  LAT_DEGREES=$(printf "%.0f" "${LATITUDE%.*}")
  LAT_MINUTES=$(echo "(${LATITUDE#${LAT_DEGREES}} * 60)" | bc -l)
  LAT_DIRECTION=$(if (( $(echo "$LATITUDE >= 0" | bc -l) )); then echo "N"; else echo "S"; fi)
  LON_DEGREES=$(printf "%.0f" "${LONGITUDE%.*}")
  LON_MINUTES=$(echo "(${LONGITUDE#${LON_DEGREES}} * 60)" | bc -l)
  LON_DIRECTION=$(if (( $(echo "$LONGITUDE >= 0" | bc -l) )); then echo "E"; else echo "W"; fi)

  # Format the NMEA GGA sentence
  RAW_NMEA_GGA="GPGGA,123519,$(printf "%02d%07.4f" ${LAT_DEGREES#-} $LAT_MINUTES),$LAT_DIRECTION,$(printf "%03d%07.4f" ${LON_DEGREES#-} $LON_MINUTES),$LON_DIRECTION,1,08,0.9,$(printf "%.1f" $ALTITUDE),M,46.9,M,,"
  CHECKSUM=$(calculate_checksum "$RAW_NMEA_GGA")
  NMEA_GGA="\$${RAW_NMEA_GGA}*${CHECKSUM}"

  # Format the VTG sentence
  RAW_NMEA_VTG="GPVTG,$(printf "%.1f" $BEARING),T,,M,$(printf "%.1f" $SPEED_KNOTS),N,$(printf "%.1f" $SPEED_KMH),K"
  CHECKSUM_VTG=$(calculate_checksum "$RAW_NMEA_VTG")
  NMEA_VTG="\$${RAW_NMEA_VTG}*${CHECKSUM_VTG}"

  # Send data via UDP
  echo "$NMEA_GGA"  
  echo "$NMEA_GGA" | socat - UDP:$SERVER_IP:$SERVER_PORT
  #echo "$NMEA_VTG"
  #echo "$NMEA_VTG" | socat - UDP:$SERVER_IP:$SERVER_PORT
  
  sleep 1
done
-----

# Pwnagotchi
main.plugins.gps_listener.enabled = true

# packages
sudo apt-get install socat
"""

class GPS(plugins.Plugin):
    __author__ = 'https://github.com/krishenriksen'
    __version__ = "1.0.0"
    __license__ = "GPL3"
    __description__ = "Receive GPS coordinates via termux-location and save whenever an handshake is captured."

    LINE_SPACING = 10
    LABEL_SPACING = 0

    def __init__(self):
        self.listen_ip = self.get_ip_address('bnep0')
        self.listen_port = "5000"
        self.write_virtual_serial = "/dev/ttyUSB1"
        self.read_virtual_serial = "/dev/ttyUSB0"
        self.baud_rate = "19200"
        self.socat_process = None
        self.pty_process = None
        self.stop_event = threading.Event()
        self.status_lock = threading.Lock()
        self.status = '-'
        self.socat_thread = threading.Thread(target=self.run_socat)
        self.agent = None
        self.last_coordinates = None

    def get_ip_address(self, interface):
        try:
            result = subprocess.run(
                ["ip", "addr", "show", interface],
                capture_output=True,
                text=True,
                check=True
            )
            for line in result.stdout.split('\n'):
                if 'inet ' in line:
                    ip_address = line.strip().split()[1].split('/')[0]
                    return ip_address
        except subprocess.CalledProcessError:
            logging.warning(f"Could not get IP address for interface {interface}")
            return None

    def set_status(self, status):
        with self.status_lock:
            self.status = status

    def get_status(self):
        with self.status_lock:
            return self.status

    def on_loaded(self):
        logging.info("GPS Listener plugin loaded")
        self.cleanup_virtual_serial_ports()
        self.create_virtual_serial_ports()
        self.socat_thread.start()

    def cleanup_virtual_serial_ports(self):
        if os.path.exists(self.write_virtual_serial):
            logging.info(f"Removing old {self.write_virtual_serial}")
            os.remove(self.write_virtual_serial)

        if os.path.exists(self.read_virtual_serial):
            logging.info(f"Removing old {self.read_virtual_serial}")
            os.remove(self.read_virtual_serial)

    def create_virtual_serial_ports(self):
        self.pty_process = subprocess.Popen(
            ["socat", "-d", "-d", f"pty,link={self.write_virtual_serial},mode=777",
             f"pty,link={self.read_virtual_serial},mode=777"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def run_socat(self):
        cmd = [
            "socat",
            f"UDP-RECVFROM:{self.listen_port},fork,reuseaddr,bind={self.listen_ip}",
            f"GOPEN:{self.write_virtual_serial}"
        ]

        while not self.stop_event.is_set():
            self.socat_process = subprocess.Popen(cmd)
            self.set_status('C')
            self.socat_process.wait()

        self.set_status('-')

    def cleanup(self):
        if self.socat_process:
            self.socat_process.terminate()
            self.socat_process.wait() # Ensure the process is reaped
        if self.pty_process:
            self.pty_process.terminate()
            self.pty_process.wait()
        self.stop_event.set()
        self.socat_thread.join()
        self.cleanup_virtual_serial_ports()

    def on_ready(self, agent):
        self.agent = agent
        if os.path.exists(self.read_virtual_serial):
            logging.info(
                f"enabling bettercap's gps module for {self.read_virtual_serial}"
            )
            try:
                agent.run("gps off")
            except Exception:
                logging.info(f"bettercap gps module was already off")
                pass

            agent.run(f"set gps.device {self.read_virtual_serial}")
            agent.run(f"set gps.baudrate {self.baud_rate}")
            agent.run("gps on")

            logging.info(f"bettercap gps module enabled on {self.read_virtual_serial}")
        else:
            self.set_status('NF')
            logging.warning("no GPS detected")

    def on_handshake(self, agent, filename, access_point, client_station):
        info = agent.session()
        coordinates = info["gps"]
        gps_filename = filename.replace(".pcap", ".gps.json")

        if coordinates and all([
            # avoid 0.000... measurements
            coordinates["Latitude"], coordinates["Longitude"]
        ]):
            self.set_status('S')
            logging.info(f"saving GPS to {gps_filename} ({coordinates})")
            with open(gps_filename, "w+t") as fp:
                json.dump(coordinates, fp)
        else:
            logging.warning("not saving GPS. Couldn't find location.")

    def on_ui_setup(self, ui):
        try:
            # Configure line_spacing
            line_spacing = int(self.options['linespacing'])
        except Exception:
            # Set default value
            line_spacing = self.LINE_SPACING

        try:
            # Configure position
            pos = self.options['position'].split(',')
            pos = [int(x.strip()) for x in pos]
            lat_pos = (pos[0] + 5, pos[1])
            lon_pos = (pos[0], pos[1] + line_spacing)
            alt_pos = (pos[0] + 5, pos[1] + (2 * line_spacing))
        except Exception:
            # Set default value based on display type
            if ui.is_waveshare_v2():
                lat_pos = (127, 74)
                lon_pos = (122, 84)
                alt_pos = (127, 94)
            elif ui.is_waveshare_v1():
                lat_pos = (130, 70)
                lon_pos = (125, 80)
                alt_pos = (130, 90)
            elif ui.is_inky():
                lat_pos = (127, 60)
                lon_pos = (122, 70)
                alt_pos = (127, 80)
            elif ui.is_waveshare144lcd():
                # guessed values, add tested ones if you can
                lat_pos = (67, 73)
                lon_pos = (62, 83)
                alt_pos = (67, 93)
            elif ui.is_dfrobot_v2():
                lat_pos = (127, 74)
                lon_pos = (122, 84)
                alt_pos = (127, 94)
            elif ui.is_waveshare27inch():
                lat_pos = (6, 120)
                lon_pos = (1, 135)
                alt_pos = (6, 150)
            elif ui.is_displayhatmini():
                lat_pos = (127, 51)
                lon_pos = (122, 61)
                alt_pos = (127, 71)
            else:
                # guessed values, add tested ones if you can
                lat_pos = (127, 51)
                lon_pos = (122, 61)
                alt_pos = (127, 71)

        with ui._lock:
            ui.add_element('gps', LabeledValue(color=BLACK, label='GPS', value='-', position=(ui.width() / 2 - 47, 0), label_font=fonts.Bold, text_font=fonts.Medium))
            ui.add_element("latitude", LabeledValue(color=BLACK, label="lat:", value="-", position=lat_pos, label_font=fonts.Small, text_font=fonts.Small, label_spacing=self.LABEL_SPACING))
            ui.add_element("longitude", LabeledValue(color=BLACK, label="long:", value="-", position=lon_pos, label_font=fonts.Small, text_font=fonts.Small, label_spacing=self.LABEL_SPACING))
            ui.add_element("altitude", LabeledValue(color=BLACK, label="alt:", value="-", position=alt_pos, label_font=fonts.Small, text_font=fonts.Small, label_spacing=self.LABEL_SPACING))

    def on_unload(self, ui):
        self.cleanup()  

        with ui._lock:
            ui.remove_element('gps')
            ui.remove_element('latitude')
            ui.remove_element('longitude')
            ui.remove_element('altitude')

    def on_ui_update(self, ui):
        ui.set('gps', self.get_status())
        if not self.agent and hasattr(ui, '_agent'):
            self.agent = ui._agent

        if self.agent:
            coordinates = self.agent.session().get('gps')
            if coordinates and coordinates.get("Latitude") is not None and coordinates.get("Longitude") is not None:
                if self.last_coordinates != coordinates:
                    ui.set("latitude", f"{coordinates['Latitude']:.4f}")
                    ui.set("longitude", f"{coordinates['Longitude']:.4f}")
                    if coordinates.get("Altitude") is not None:
                        ui.set("altitude", f"{coordinates['Altitude']:.1f}m")
                    self.last_coordinates = coordinates
