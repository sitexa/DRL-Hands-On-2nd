import platform

import torch


def get_device(gpu=True):
    if not gpu:
        return "cpu"
    sys_platform = platform.platform().lower()
    if "macos" in sys_platform:
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


if __name__ == "__main__":
    print(platform.platform().lower())
    print(get_device())
