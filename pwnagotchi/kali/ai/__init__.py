import os
import time
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def load(config, agent, epoch, from_disk=True):
    config = config['ai']
    if not config['enabled']:
        logging.info("ai disabled")
        return False

    try:
        begin = time.time()
        logging.info("[ai] bootstrapping dependencies ...")

        SB_BACKEND = "stable_baselines3"
        start = time.time()
        try:
            from stable_baselines3 import A2C
            from stable_baselines3.a2c import MlpPolicy
            from stable_baselines3.common.vec_env import DummyVecEnv
            SB_A2C_POLICY = MlpPolicy
            logging.debug("[ai] stable_baselines3 loaded in %.2fs", time.time() - start)
            for key in ['alpha', 'epsilon', 'lr_schedule']:
                if key in config['params']:
                    del config['params'][key]
        except Exception:
            from stable_baselines import A2C
            from stable_baselines.common.policies import MlpLstmPolicy
            from stable_baselines.common.vec_env import DummyVecEnv
            SB_BACKEND = "stable_baselines"
            SB_A2C_POLICY = MlpLstmPolicy
            logging.debug("[ai] stable_baselines loaded in %.2fs", time.time() - start)

        import pwnagotchi.kali.ai.gym as wrappers
        env = wrappers.Environment(agent, epoch)
        env = DummyVecEnv([lambda: env])

        a2c = A2C(SB_A2C_POLICY, env, **config['params'])

        # Prefer trainer-selected path (tool brain), fallback to config for compatibility.
        primal_requested = bool(config.get('primal', False))
        nn_path = getattr(agent, '_nn_path', None) or config.get('path')
        if primal_requested and nn_path:
            nn_path = os.path.join(os.path.dirname(nn_path), 'primal.nn')
        if primal_requested and not wrappers.PRIMAL:
            logging.warning("[ai] ai.primal=true but gym wrapper is not in primal mode; keeping primal.nn path but runtime may be legacy gym")

        if from_disk and nn_path and os.path.exists(nn_path):
            try:
                size = os.path.getsize(nn_path)
                logging.info("[ai] loading %s ...", nn_path)
                logging.info("[ai] model file size: %.2f MB", size / (1024.0 * 1024.0))
                load_start = time.time()
                a2c = a2c.load(nn_path, env)
                logging.info("[ai] model loaded in %.2fs", time.time() - load_start)
            except Exception as load_exc:
                logging.warning("[ai] could not load %s (%s), creating fresh model", nn_path, load_exc)
                logging.info("[ai] model created with params: %s", config['params'])
        else:
            logging.info("[ai] model created with params: %s", config['params'])

        logging.debug("[ai] total loading time %.2fs", time.time() - begin)
        return a2c
    except Exception as e:
        logging.exception("error while starting AI (%s)", e)

    logging.warning("[ai] AI not loaded!")
    return False
