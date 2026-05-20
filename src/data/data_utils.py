import pandas as pd
import random
import torch
from datasets import Dataset, concatenate_datasets
import re

def load_data(args, split):
    if args.data == 'pro_medicine' :
        from data.mmlu_pro_medicine import load_data 
        return load_data(args, split=split)
    elif args.data == 'formal_logic' :
        from data.mmlu_formal_logic import load_data 
        return load_data(args, split=split)
    elif args.data == 'gsm8k' :
        from data.gsm8k import load_data as load_gsm8k
        return load_gsm8k(args, split=split)


