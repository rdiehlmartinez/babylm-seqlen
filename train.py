import os
import time

import wandb
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    OPTConfig,
    OPTForCausalLM,
    Trainer,
    TrainingArguments,
)

# Constants and Helpers
LOG_EVERY_N_STEPS = 10
SAVE_EVERY_N_STEPS = 500
TRAIN_EPOCHS = 10


def get_deepspeed_config():
    return {
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {"device": "cpu"},
            "offload_param": {"device": "cpu"},
        },
        "train_batch_size": 64,
        "gradient_accumulation_steps": 1,
        "fp16": {"enabled": True},
    }


# Model Train Script


def train_model(
    model_type="opt", seq_len=128, use_deepspeed=False, push_to_hub=True, dry_run=False
):
    dataset = load_dataset(f"babylm-seqlen/train_100M_{seq_len}_single_shuffle")
    dataset = dataset.map(lambda x: {"labels": x["input_ids"]})

    train_dataset = dataset["train"]

    if dry_run:
        train_dataset = train_dataset.select(range(100))
        output_dir = f"./dryruns/{model_type}-babylm-{seq_len}"
    else:
        output_dir = f"./checkpoints/{model_type}-babylm-{seq_len}"

    os.makedirs(output_dir, exist_ok=True)

    run_name = f"{model_type}_babylm_{seq_len}"

    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m", use_fast=True)
    tokenizer.model_max_length = seq_len

    if model_type == "opt":
        config = OPTConfig(
            vocab_size=50257,
            hidden_size=768,
            num_attention_heads=12,
            num_hidden_layers=12,
            intermediate_size=3072,
            max_position_embeddings=2048,
            torch_dtype="float16",
        )
        model = OPTForCausalLM(config)

    elif model_type == "mamba":
        from transformers import MambaConfig, MambaForCausalLM

        config = MambaConfig(
            vocab_size=50257,
            hidden_size=256,
            num_hidden_layers=6,
            intermediate_size=1024,  # Adjust as needed
        )
        model = MambaForCausalLM(config)

    wandb.init(
        project="babylm-seqlen",
        name=run_name,
        config={
            "model_type": model_type,
            "seq_len": seq_len,
            "dry_run": dry_run,
        },
        mode="disabled" if dry_run else "online",
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=64,  # TODO: tune this
        gradient_accumulation_steps=4,  # TODO: tune this
        num_train_epochs=TRAIN_EPOCHS,
        evaluation_strategy="no",
        save_strategy="steps",
        save_steps=SAVE_EVERY_N_STEPS,
        bf16=True,
        report_to="wandb",
        run_name=run_name,
        deepspeed=get_deepspeed_config() if use_deepspeed else None,
        logging_steps=LOG_EVERY_N_STEPS,
        disable_tqdm=False,
        push_to_hub=push_to_hub,
        hub_model_id=f"babylm-seqlen/{model_type}-{seq_len}",
        hub_strategy="every_save",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    start_time = time.time()
    trainer.train()
    end_time = time.time()

    print(
        f"✅ Training {model_type.upper()} for seq_len {seq_len} done in {end_time - start_time:.2f}s"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type", type=str, default="opt", choices=["opt", "mamba"]
    )
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--use_deepspeed", action="store_true")
    parser.add_argument(
        "--no_push_to_hub",
        action="store_true",
        help="If set, do NOT push to the Hugging Face Hub.",
    )
    parser.add_argument("--dry_run", action="store_true")

    args = parser.parse_args()

    train_model(
        model_type=args.model_type,
        seq_len=args.seq_len,
        use_deepspeed=args.use_deepspeed,
        push_to_hub=not args.no_push_to_hub,
        dry_run=args.dry_run,
    )
