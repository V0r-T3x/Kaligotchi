import json
import logging
import requests
import time

import websockets
import websockets.exceptions
import asyncio

from requests.auth import HTTPBasicAuth


def decode(r, verbose_errors=True):
    try:
        return r.json()
    except Exception as e:
        if r.status_code == 200:
            logging.error("error while decoding json: error='%s' resp='%s'" % (e, r.text))
        else:
            err = "error %d: %s" % (r.status_code, r.text.strip())
            if verbose_errors:
                logging.info(err)
                # moved the raise under here..  it was even with "if"
                raise Exception(err)
        return r.text


class Client(object):
    def __init__(self, hostname='localhost', scheme='http', port=8081, username='user', password='pass'):
        self.hostname = hostname
        self.scheme = scheme
        self.port = port
        self.username = username
        self.password = password
        self.url = "%s://%s:%d/api" % (scheme, hostname, port)
        self.websocket = "ws://%s:%s@%s:%d/api" % (username, password, hostname, port)
        self.auth = HTTPBasicAuth(username, password)
        self._connection_errors = 0

    # session takes optional argument to pull a sub-dictionary
    #  ex.: "session/wifi", "session/ble"
    def session(self, sess="session"):
        try:
            r = requests.get("%s/%s" % (self.url, sess), auth=self.auth, timeout=30)
            return decode(r)
        except Exception as e:
            logging.error("error while fetching session: %s" % e)
            self._connection_errors += 1
            return None

    def wait_until_ready(self, timeout=60, interval=2):
        """Blocks until the Bettercap API is reachable or timeout is reached."""
        start_time = time.time()
        logging.info("Waiting for Bettercap API to breathe...")
        while time.time() - start_time < timeout:
            try:
                # A simple session call to check if the heart is beating
                self.session()
                logging.info("Bettercap is alive.")
                return True
            except Exception:
                time.sleep(interval)
        logging.error("Bettercap failed to initialize in time.")
        return False

    async def start_websocket(self, consumer):
        s = "%s/events" % self.websocket
        backoff = 2  # Start with 2 seconds
        max_backoff = 60 # Cap it at a minute

        while True:
            try:
                async with websockets.connect(s, open_timeout=10, ping_interval=60, ping_timeout=90) as ws:
                    backoff = 2  # Reset backoff on successful connection
                    async for msg in ws:
                        try:
                            await consumer(msg)
                        except Exception as ex:
                            logging.error("Error while parsing event (%s)", ex)
            except Exception as e:
                self._connection_errors += 1
                logging.warning(f"Bettercap connection failed: {e}. Retrying in {backoff}s...")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff) # Exponentially increase wait time

    def run(self, command, verbose_errors=True):
        try:
            r = requests.post("%s/session" % self.url, auth=self.auth, json={'cmd': command}, timeout=30)
            try:
                return decode(r, verbose_errors=verbose_errors)
            except Exception as e:
                if "wifi is not running" in ("%s" % e):
                    try:
                        self.run("wifi.recon on")
                    except Exception as e2:
                        logging.exception("%s, after %s decoding: %s" % (e2, e, repr(r)))
                elif "is an unknown BSSID" in ("%s" % e):
                    logging.debug(e)
                    raise   # back to assoc or deauth
                else:
                    logging.exception(e)
        except Exception as e:
            logging.error("error while running command %s: %s" % (command, e))
            self._connection_errors += 1
            return None
