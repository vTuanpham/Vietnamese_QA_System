import sys
import os
import gc
import time
import random
import argparse
sys.path.insert(0,r'./')
from functools import wraps

import torch
import torch.distributed as dist
import numpy as np


def dist_print(string: str):
    if dist.is_initialized():
        # Running in a distributed setting (multiple processes)
        if dist.get_rank() == 0:
            # Main process (rank 0)
            print(string)
    else:
        # Running in a single-process setting
        print(string)


def in_notebook():
    """
    Returns ``True`` if the module is running in IPython kernel,
    ``False`` if in IPython shell or other Python shell.
    """
    return 'ipykernel' in sys.modules


def set_seed(value):
    print("\n Random Seed: ", value)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    torch.use_deterministic_algorithms(True, warn_only=True)
    np.random.seed(value)


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} Took {total_time:.4f} seconds')

        return result
    return timeit_wrapper
