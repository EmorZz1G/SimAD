import random
import warnings
import torch
import torch.backends.cudnn as cudnn


def setup_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    warnings.warn(
        "You have chosen to seed training. "
        "This will turn on the CUDNN deterministic setting, "
        "which can slow down your training considerably! "
        "You may see unexpected behavior when restarting "
        "from checkpoints."
    )