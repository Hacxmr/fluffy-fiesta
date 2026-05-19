import pandas as pd
import random
import torch
from datasets import Dataset, concatenate_datasets
import re
from data.gsm8k import load_data as load_gsm8k

def load_data(args, split):
    return load_gsm8k(args, split=split)