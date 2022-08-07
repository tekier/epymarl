import numpy as np
import os
import collections
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th
from utils.logging import get_logger
import yaml
import pathlib

from run import run

from griddly import GymWrapperFactory

SETTINGS['CAPTURE_MODE'] = "fd"  # set to "no" if you want to see stdout/stderr in console
logger = get_logger()

ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

root_dir = pathlib.Path(__file__).parent.resolve().parent.resolve()
results_path = os.path.join(root_dir, "results")


@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    config = config_copy(_config)
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config['env_args']['seed'] = config["seed"]

    # run the framework
    run(_run, config, _log)


def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


def register_griddly_env(_env_name):
    gdy_path = os.path.join(root_dir, 'smac_lite/gdy/5m_vs_6m_lite.yml')
    wrapper = GymWrapperFactory()
    wrapper.build_gym_from_yaml(_env_name, gdy_path)


if __name__ == '__main__':
    params = deepcopy(sys.argv)
    th.set_num_threads(1)

    env_name = f'5m-vs-6m-lite'
    register_griddly_env(env_name)

    # Get the defaults from default.yaml
    with open(os.path.join(root_dir, "src/config/default.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--config", "algs")
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)

    try:
        map_name = config_dict["env_args"]["map_name"]
    except KeyError:
        map_name = config_dict["env_args"]["key"]

    # now add all the config to sacred
    ex.add_config(config_dict)

    for param in params:
        if param.startswith("env_args.map_name"):
            map_name = param.split("=")[1]
        elif param.startswith("env_args.key"):
            map_name = param.split("=")[1]

    if map_name is None:
        map_name = os.environ.get('MAP_NAME')

    # Save to disk by default for sacred
    logger.info("Saving to FileStorageObserver in results/sacred.")
    file_obs_path = os.path.join(results_path, f"sacred/{config_dict['name']}/{map_name}")
    # TODO: FileStorageObserver.create( ... ) is deprecated - remove!!!
    ex.observers.append(FileStorageObserver.create(file_obs_path))
    ex.run_commandline(params)

