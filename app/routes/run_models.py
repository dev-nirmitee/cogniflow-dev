import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import numpy as np
import pandas as pd
import argparse
from guarvis_models import MMOE_stress, ConvNet1D_stress
from thop import profile
import pickle
import time
import warnings
warnings.filterwarnings("ignore")

def set_random_seed(seed):
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.manual_seed(seed)

set_random_seed(4)


def run_imh_model(input_data):
    result = {
        "imh_depression": "low",
        "imh_anxiety": "medium"}
    return result

def run_guarvis_model(input_data):
    device = torch.device("cpu")
    model = MMOE_stress(input_dim=64, input_channel=3, num_experts=3, hidden_dim=128, num_class=4).to(device)
    output = model(torch.randn(1, 3, 64).to(device))
    required_output = output[:6]
    categorical_mapping = {0: "low", 1: "low", 2: "medium", 3: "high"}
    result_keys = ['guarvis_calm', 'guarvis_depression', 'guarvis_sleep', 'guarvis_hope', 'guarvis_think', 'guarvis_stress']
    result = {k: categorical_mapping[v] for k, v in zip(result_keys, required_output)}
    return result

