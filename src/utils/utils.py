import sys
import os
import gc
import time
import random
import argparse
sys.path.insert(0,r'./')
from functools import wraps

import torch
import numpy as np


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
