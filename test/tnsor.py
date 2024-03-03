import torch
import torch.nn as nn
import numpy as np

device="cpu"
state = np.array([1, 2, 3, 4, 5])

state_a = np.array([state], copy=False)
state_v = torch.tensor(state_a).to(device)
