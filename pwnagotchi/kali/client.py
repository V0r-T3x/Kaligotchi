import asyncio
import logging
import time
from typing import Any, Dict, Optional

import requests
import websockets
from requests.auth import HTTPBasicAuth


def decode(response, verbose_errors=True):
    try:
        return response.json()
    except Exception as exc:
        if response.status_code == 200:
            logging.error("error decoding json: error='%s' resp='%s'", exc, response.text)
        else:
            err = "error %d: %s" % (response.status_code, response.text.strip())
            if verbose_errors:
                logging.info(err)
                raise Exception(err)
        return response.text


class ApiClient(object):
    """Generic HTTP/WebSocket API client for toolbox adapters."""

    def __init__(
        self,
        hostname='localhost',
        scheme='http',
        port=8081,
        username='user',
        password='pass',
        api_base_path='/api',
        command_endpoint='session',
        command_field='cmd',
        websocket_events_path='events',
        request_timeout=30,
    ):
        self.hostname = hostname
        self.scheme = scheme
        self.port = int(port)
        self.username = username
        self.password = password

        self.api_base_path = str(api_base_path or '/api').strip()
        if not self.api_base_path.startswith('/'):
            self.api_base_path = '/' + self.api_base_path

        self.command_endpoint = command_endpoint
        self.command_field = command_field
        self.websocket_events_path = websocket_events_path
        self.request_timeout = int(request_timeout)

        self.auth = HTTPBasicAuth(username, password)
        self._connection_errors = 0
        self.running = True

        self.url = f"{self.scheme}://{self.hostname}:{self.port}{self.api_base_path}"
        self.websocket = self._build_websocket_base_url()

    def _build_websocket_base_url(self):
        ws_scheme = 'wss' if self.scheme == 'https' else 'ws'
        return f"{ws_scheme}://{self.username}:{self.password}@{self.hostname}:{self.port}{self.api_base_path}"

    def _join(self, base: str, path: str) -> str:
        path = str(path or '').strip('/')
        if not path:
            return base
        return f"{base}/{path}"

    def _probe_ready(self, path='session') -> bool:
        """Lightweight readiness probe without noisy logs."""
        try:
            endpoint = self._join(self.url, path)
            response = requests.get(endpoint, auth=self.auth, timeout=self.request_timeout)
            return response.status_code == 200
        except Exception:
            return False

    def get_json(self, path='session'):
        try:
            endpoint = self._join(self.url, path)
            response = requests.get(endpoint, auth=self.auth, timeout=self.request_timeout)
            return decode(response)
        except Exception as exc:
            logging.error("api get failed (%s): %s", path, exc)
            self._connection_errors += 1
            return None

    def post_json(self, path='session', payload: Optional[Dict[str, Any]] = None, verbose_errors=True):
        try:
            endpoint = self._join(self.url, path)
            response = requests.post(endpoint, auth=self.auth, json=(payload or {}), timeout=self.request_timeout)
            return decode(response, verbose_errors=verbose_errors)
        except Exception as exc:
            logging.error("api post failed (%s): %s", path, exc)
            self._connection_errors += 1
            return None

    def session(self, sess='session'):
        return self.get_json(sess)

    def run(self, command, verbose_errors=True):
        return self.post_json(
            path=self.command_endpoint,
            payload={self.command_field: command},
            verbose_errors=verbose_errors,
        )

    def close(self):
        self.running = False

    def wait_until_ready(self, timeout=60, interval=2, check_path='session'):
        start = time.time()
        logging.info("waiting for API readiness at %s ...", self.url)

        while time.time() - start < timeout:
            if self._probe_ready(check_path):
                logging.info("API is ready.")
                return True
            time.sleep(interval)

        logging.error("API readiness timed out after %ss.", timeout)
        return False

    async def start_websocket(self, consumer, events_path=None, status_callback=None):
        events_path = events_path or self.websocket_events_path
        ws_url = self._join(self.websocket, events_path)

        backoff = 2
        max_backoff = 60

        while self.running:
            try:
                async with websockets.connect(ws_url, open_timeout=10, ping_interval=60, ping_timeout=90) as ws:
                    if callable(status_callback):
                        try:
                            status_callback(True, 'websocket')
                        except Exception:
                            logging.debug("websocket status callback failed on connect", exc_info=True)
                    backoff = 2
                    async for msg in ws:
                        if not self.running:
                            break
                        try:
                            await consumer(msg)
                        except Exception as exc:
                            logging.error("websocket consumer error: %s", exc)
            except Exception as exc:
                if not self.running:
                    break
                self._connection_errors += 1
                if callable(status_callback):
                    try:
                        status_callback(False, 'websocket')
                    except Exception:
                        logging.debug("websocket status callback failed on disconnect", exc_info=True)
                logging.warning("websocket connection failed: %s. retrying in %ss", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)


# Backward-compatible alias.
Client = ApiClient
