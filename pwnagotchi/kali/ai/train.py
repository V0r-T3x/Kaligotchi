import _thread
import threading
import time
import random
import os
import json
import logging
import prctl

import pwnagotchi.plugins as plugins
import pwnagotchi.kali.ai as ai
import pwnagotchi.kali.ai.gym as ai_gym


class Stats(object):
    def __init__(self, path, events_receiver, primal=False):
        self._lock = threading.Lock()
        self._receiver = events_receiver
        self.primal = primal

        self.path = path
        self.born_at = time.time()
        # total epochs lived (trained + just eval)
        self.epochs_lived = 0
        # total training epochs
        self.epochs_trained = 0

        self.worst_reward = 0.0
        self.best_reward = 0.0

        self.load()

    def on_epoch(self, data, training):
        best_r = False
        worst_r = False
        with self._lock:
            reward = data['reward']
            if reward < self.worst_reward:
                self.worst_reward = reward
                worst_r = True

            elif reward > self.best_reward:
                best_r = True
                self.best_reward = reward

            self.epochs_lived += 1
            if training:
                self.epochs_trained += 1

        if best_r or worst_r or (self.epochs_lived % 10 == 0):
            self.save()

        if best_r:
            self._receiver.on_ai_best_reward(reward)
        elif worst_r:
            self._receiver.on_ai_worst_reward(reward)

    def load(self):
        with self._lock:
            if os.path.exists(self.path) and os.path.getsize(self.path) > 0:
                logging.info("[ai] loading %s" % self.path)
                with open(self.path, 'rt') as fp:
                    obj = json.load(fp)

                self.born_at = obj['born_at']
                self.epochs_lived, self.epochs_trained = obj['epochs_lived'], obj['epochs_trained']
                self.best_reward, self.worst_reward = obj['rewards']['best'], obj['rewards']['worst']

    def save(self):
        with self._lock:
            logging.debug("[ai] saving %s" % self.path)

            data = json.dumps({
                'born_at': self.born_at,
                'epochs_lived': self.epochs_lived,
                'epochs_trained': self.epochs_trained,
                'rewards': {
                    'best': self.best_reward,
                    'worst': self.worst_reward
                }
            })

            temp = "%s.tmp" % self.path
            back = "%s.bak" % self.path
            with open(temp, 'wt') as fp:
                fp.write(data)

            if os.path.isfile(self.path):
                os.replace(self.path, back)
            os.replace(temp, self.path)


class AsyncTrainer(object):
    @staticmethod
    def _resolve_kali_brains_root(config):
        kali_cfg = config.get('kali', {}) if isinstance(config, dict) else {}
        brains_cfg = kali_cfg.get('brains', {}) if isinstance(kali_cfg, dict) else {}
        root = brains_cfg.get('path', '/root/brains') if isinstance(brains_cfg, dict) else '/root/brains'
        return str(root or '/root/brains')

    def __init__(self, config):
        self._config = config
        self._model = None
        self._is_training = False
        self._training_epochs = 0
        self._ai_pause = False
        self._primal_enabled = bool(self._config.get('ai', {}).get('primal', False))
        
        self._brains_root_path = self._resolve_kali_brains_root(self._config)

        self._nn_path = os.path.join(self._brains_root_path, 'default', 'brain.nn')
        if self._primal_enabled:
            primal_nn_path = os.path.join(os.path.dirname(self._nn_path), 'primal.nn')
            #if os.path.exists(primal_nn_path):
            #    os.remove(primal_nn_path)
            self._nn_path = primal_nn_path

        os.makedirs(os.path.dirname(self._nn_path), exist_ok=True)
        self._stats = Stats(os.path.join(os.path.dirname(self._nn_path), 'brain.json'), self, self._primal_enabled)
        if self._primal_enabled and not ai_gym.PRIMAL:
            logging.warning("[ai] ai.primal=true but gym wrapper did not enter primal mode; model path still forced to primal.nn")

    def configure_brain_path(self, nn_path):
        if not nn_path:
            return

        self._nn_path = nn_path
        if self._primal_enabled:
            self._nn_path = os.path.join(os.path.dirname(self._nn_path), 'primal.nn')
        os.makedirs(os.path.dirname(self._nn_path), exist_ok=True)
        self._stats = Stats(os.path.join(os.path.dirname(self._nn_path), 'brain.json'), self, self._primal_enabled)

    def set_training(self, training, for_epochs=0):
        self._is_training = training
        self._training_epochs = for_epochs

        if training:
            plugins.on('ai_training_start', self, for_epochs)
        else:
            plugins.on('ai_training_end', self)

    def is_training(self):
        return self._is_training

    def training_epochs(self):
        return self._training_epochs

    def start_ai(self):
        _thread.start_new_thread(self._ai_worker, ())

    def _save_ai(self):
        logging.info("[ai] saving model to %s ..." % self._nn_path)
        os.makedirs(os.path.dirname(self._nn_path), exist_ok=True)
        temp = "%s.tmp" % self._nn_path
        self._model.save(temp)
        os.replace(temp, self._nn_path)

    def on_ai_step(self):
        # Avoid calling VecEnv.render() from inside env.step() execution path.
        # Rendering is handled by on_ai_training_step callback.
        self._stats.on_epoch(self._epoch.data(), self._is_training)

    def on_ai_training_step(self, _locals, _globals):
        self._model.env.render()
        plugins.on('ai_training_step', self, _locals, _globals)
        # SB3 callback must return True to keep training.
        return True

    def on_ai_policy(self, new_params):
        # Get bias from the environment
        try:
            bias = self._model.env.get_attr('last')[0].get('reflex_bias', {})
        except Exception:
            bias = {}

        # Shape the policy: e.g., apply the recon_time_multiplier
        if 'recon_time' in new_params and 'recon_time_multiplier' in bias:
            new_params['recon_time'] *= bias['recon_time_multiplier']

        # Apply the offset to RSSI
        if 'min_rssi' in new_params and 'min_rssi_offset' in bias:
            new_params['min_rssi'] += bias['min_rssi_offset']

        # Throttle interactions if the interface is choking
        if 'max_interactions' in new_params and 'interaction_scale' in bias:
            new_params['max_interactions'] = max(1, int(new_params['max_interactions'] * bias['interaction_scale']))

        # Apply TTL multiplier from reflex
        if 'ttl_multiplier' in bias:
            if 'ap_ttl' in new_params:
                new_params['ap_ttl'] = int(new_params['ap_ttl'] * bias['ttl_multiplier'])
            if 'sta_ttl' in new_params:
                new_params['sta_ttl'] = int(new_params['sta_ttl'] * bias['ttl_multiplier'])

        plugins.on('ai_policy', self, new_params)
        logging.info("[ai] setting new policy:")
        for name, value in new_params.items():
            if name in self._config['personality']:
                if name == 'channels':
                    value = list(filter(lambda x: x in self._allowed_channels, value))
                curr_value = self._config['personality'][name]
                if curr_value != value:
                    logging.info("[ai] ! %s: %s -> %s" % (name, curr_value, value))
                    self._config['personality'][name] = value
            else:
                logging.error("[ai] param %s not in personality configuration!" % name)

        self.run('set wifi.ap.ttl %d' % self._config['personality']['ap_ttl'])
        self.run('set wifi.sta.ttl %d' % self._config['personality']['sta_ttl'])
        self.run('set wifi.rssi.min %d' % self._config['personality']['min_rssi'])

    def on_ai_ready(self):
        self._view.on_ai_ready()
        if hasattr(self, '_refresh_ui_mode_label'):
            self._refresh_ui_mode_label()
            logging.info("[kali] AI ready -> mode badge K-AI")
        plugins.on('ai_ready', self)

    def _set_training_mode_badge(self, legacy_label):
        if hasattr(self, 'mode') and getattr(self, 'mode', None) == 'kali' and hasattr(self, '_refresh_ui_mode_label'):
            logging.info("[kali] suppressed legacy %s mode badge write", legacy_label.strip())
            self._refresh_ui_mode_label()
            return
        self._view.set("mode", legacy_label)

    def on_ai_best_reward(self, r):
        logging.info("[ai] best reward so far: %s" % r)
        self._view.on_motivated(r)
        plugins.on('ai_best_reward', self, r)

    def on_ai_worst_reward(self, r):
        logging.info("[ai] worst reward so far: %s" % r)
        self._view.on_demotivated(r)
        plugins.on('ai_worst_reward', self, r)

    def _ai_worker(self):
        prctl.set_name("ai worker")
        self._model = ai.load(self._config, self, self._epoch)

        if self._model:
            if not os.path.exists(self._nn_path):
                self._save_ai()

            self.on_ai_ready()
            prctl.set_name("ai: ready")

            epochs_per_episode = self._config['ai']['epochs_per_episode']

            obs = None
            force_initial_training = True
            while True:
                if self._ai_pause:
                    time.sleep(0.1)
                    continue
                self._model.env.render()
                # enter in training mode?
                should_train = force_initial_training or (random.random() > self._config['ai']['laziness'])
                if should_train:
                    if force_initial_training:
                        logging.info("[ai] forcing initial training cycle")
                    logging.info("[ai] learning for %d epochs ..." % epochs_per_episode)
                    prctl.set_name("ai: training")
                    try:
                        self.set_training(True, epochs_per_episode)
                        # back up brain file before starting new training set
                        if os.path.isfile(self._nn_path):
                            back = "%s.bak" % self._nn_path
                            os.replace(self._nn_path, back)
                        self._set_training_mode_badge("  ai")
                        self._model.learn(total_timesteps=epochs_per_episode, callback=self.on_ai_training_step)
                        self._save_ai()
                        self._set_training_mode_badge("  AI")
                    except Exception as e:
                        logging.exception("[ai] error while training (%s)", e)
                    finally:
                        force_initial_training = False
                        self.set_training(False)
                        prctl.set_name("ai: pwning")
                        try:
                            obs = self._model.env.reset()
                        except Exception as e:
                            logging.exception("[ai] env.reset() failed after training (%s)", e)
                            obs = None
                # init the first time
                elif obs is None:
                    logging.info("[ai] skipping training this cycle (laziness gate), running inference")
                    try:
                        obs = self._model.env.reset()
                    except Exception as e:
                        logging.exception("[ai] env.reset() failed (%s)", e)
                        obs = None

                # run the inference
                if obs is not None:
                    try:
                        action, _ = self._model.predict(obs)
                        step_out = self._model.env.step(action)
                        if isinstance(step_out, tuple):
                            obs = step_out[0]
                        else:
                            obs = None
                    except Exception as e:
                        logging.exception("[ai] inference step failed (%s)", e)
                        obs = None
