import torch
import json
import os
from torch.nn import functional as F
from transformers import Trainer, DataCollatorForLanguageModeling
class CustomDataCollator(DataCollatorForLanguageModeling):
    def __call__(self, features):
        for f in features:
            f.pop("attention_mask", None)
        return super().__call__(features)
