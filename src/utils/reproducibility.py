import os
import torch
import random
import numpy as np

def set_seed(seed: int):

    if seed < 0:
        raise ValueError("The seed must be non-negative.")

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Improves reproducibility, but may reduce speed.
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    except ImportError:
        pass
