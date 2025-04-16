import torch
import json
import os
from torch.nn import functional as F
from transformers import Trainer, DataCollatorForLanguageModeling
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel, MambaConfig

class MambaTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        logits = model(input_ids).logits
        loss = F.cross_entropy(
            logits[..., :-1, :].contiguous().view(-1, logits.size(-1)),
            labels[..., 1:].contiguous().view(-1),
            ignore_index=-100,
        )
        return (loss, logits) if return_outputs else loss

def save_mamba_model(model, config, output_dir, tokenizer=None):
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config.__dict__, f, indent=2)
    if tokenizer:
        tokenizer.save_pretrained(output_dir)
