import torch
import yaml
from functools import wraps
import time

import structlog

logger = structlog.getLogger(__name__)


def set_device(mps: bool = False, cuda: bool = False):
    """
    Set the device to cuda and default tensor types to FloatTensor on the device
    """
    device = "cpu"
    if torch.cuda.is_available() and cuda:
        device = "cuda"
    elif torch.backends.mps.is_available() and mps:
        device = "mps"

    torch_device = torch.device(device)
    return torch_device


def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def time_job(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        start = time.time()
        function_result = function(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(
            "{} {}: {:0.3f} min".format(
                args[0].__class__.__name__, function.__name__, elapsed / 60
            )
        )
        return function_result
    return wrapped
