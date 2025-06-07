import os
import time

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    OPTConfig,
    OPTForCausalLM,
    Trainer,
    TrainingArguments,
)

from transformers.trainer_utils import (
    SaveStrategy,
)

from transformers.trainer_callback import (
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
    
import wandb

# --- Constants and Helpers --- #

TRAIN_EPOCHS = 10
GLOBAL_BATCH_SIZE = 64  # NOTE: (rdm) 64 is nice because 64*16k = 1M tokens per batch

def get_deepspeed_config(accumulation_steps, num_devices):
    return {
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {"device": "cpu"},
            "offload_param": {"device": "cpu"},
        },
        "train_batch_size": GLOBAL_BATCH_SIZE // num_devices,
        "gradient_accumulation_steps": accumulation_steps,
        "bf16": {"enabled": True},
    }

class CustomCheckpointingCallback(TrainerCallback):
    """
    Implements BabyLM checkpointing:
    - Every 1M words until 10M
    - Every 10M words until 100M
    - Every 100M words until 1B
    """
    def __init__(self, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.words_per_step = GLOBAL_BATCH_SIZE * seq_len
        # Build the list of checkpoint word counts
        self.checkpoint_words = (
            [i * 1_000_000 for i in range(1, 11)] +      # 1M to 10M
            [i * 10_000_000 for i in range(2, 11)] +     # 20M to 100M
            [i * 100_000_000 for i in range(2, 11)]      # 200M to 1B
        )
        self.next_checkpoint_idx = 0

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        words_seen = state.global_step * self.words_per_step
        if (self.next_checkpoint_idx < len(self.checkpoint_words) and
            words_seen >= self.checkpoint_words[self.next_checkpoint_idx]):
            control.should_save = True
            print(f"Checkpointing at {self.checkpoint_words[self.next_checkpoint_idx]:,} words (step {state.global_step})")
            self.next_checkpoint_idx += 1
        return control

# --- Model Train Script --- #

def train_model(
    model_type="opt",
    seq_len=128,
    use_deepspeed=False,
    push_to_hub=True,
    dry_run=False,
    num_devices=4,
    accumulation_steps=1,
    use_warmup=False,
):
    ###
    ### Setup Dataset and Models
    ###

    per_device_batch_size = GLOBAL_BATCH_SIZE / (accumulation_steps * num_devices)
    if int(per_device_batch_size) != per_device_batch_size:
        raise ValueError(
            f"Batch size {per_device_batch_size} is not an integer. "
            f"Please adjust the GLOBAL_BATCH_SIZE, num_devices, and accumulation_steps."
        )
    per_device_batch_size = int(per_device_batch_size)
    print(f"Per device batch size: {per_device_batch_size} for an effective batch size of {accumulation_steps} * {num_devices} = {GLOBAL_BATCH_SIZE}")

    try:
        dataset = load_dataset(f"babylm-seqlen/train_100M_{seq_len}_single_shuffle")
    except Exception as e:
        print(f"Dataset for seq_len {seq_len} not found.")
        print(f"Error: {e}")
        exit(1)
    dataset = dataset.map(lambda x: {"labels": x["input_ids"]}, num_proc=16)

    train_dataset = dataset["train"]

    if dry_run:
        train_dataset = train_dataset.select(range(100))
        output_dir = f"./dryruns/{model_type}-babylm-{seq_len}" + ("-warmup" if use_warmup else "")
    else:
        output_dir = f"./checkpoints/{model_type}-babylm-{seq_len}" + ("-warmup" if use_warmup else "")

    os.makedirs(output_dir, exist_ok=True)

    run_name = f"{model_type}_babylm_{seq_len}" + ("_warmup" if use_warmup else "")

    if model_type == "opt":
        config = OPTConfig(
            vocab_size=50257,
            hidden_size=768,
            num_attention_heads=12,
            num_hidden_layers=12,
            ffn_dim=3072,
            max_position_embeddings=seq_len,
        )
        model = OPTForCausalLM(config)

    elif model_type == "mamba":
        from transformers import MambaConfig, MambaForCausalLM

        config = MambaConfig(
            vocab_size=50257,
            hidden_size=768,
            num_hidden_layers=32,
        )
        model = MambaForCausalLM(config)

    local_rank = int(os.environ.get("RANK", 0))
    if local_rank == 0:
        wandb.init(
            entity="babylm-seqlen",
            project=f"{model_type}-models",
            name=run_name,
            mode="disabled" if dry_run else "online",
        )

    ###
    ### Setup Training Arguments
    ###

    # Initial checkpointing rate is every 1M words, which is 1% of an epoch.
    # We then increase the checkpointing rate by a factor of 10 every 10 checkpoints.
    total_steps = TRAIN_EPOCHS * len(train_dataset) // GLOBAL_BATCH_SIZE
    initial_save_steps = max(1, total_steps//1000)
    warmup_steps = int(total_steps * 0.05) if use_warmup else 0  # 5% of total steps for warmup

    custom_checkpointing_callback = CustomCheckpointingCallback(seq_len)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=accumulation_steps,
        num_train_epochs=TRAIN_EPOCHS,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=initial_save_steps,
        bf16=True,
        report_to="wandb",
        run_name=run_name,
        deepspeed=get_deepspeed_config(accumulation_steps, num_devices) if use_deepspeed else None,
        logging_steps=max(total_steps // 1000, 1),
        disable_tqdm=False,
        push_to_hub=push_to_hub,
        hub_model_id=f"babylm-seqlen/{model_type}-{seq_len}" + ("-warmup" if use_warmup else ""),
        hub_strategy="every_save",
        learning_rate=5e-5*(seq_len/64) if use_warmup else 5e-5,  # Scale learning rate with sequence length if using warmup
        warmup_steps=warmup_steps,  # Add warmup steps if enabled
        lr_scheduler_type="linear" if use_warmup else "constant"  # Use linear warmup if enabled
    )

    print(f"Training arguments:\n{training_args}")

    ###
    ### Setup Trainer
    ###

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        callbacks=[custom_checkpointing_callback]
    )

    if push_to_hub:
        # pushing up tokenizer to hub
        tokenizer = AutoTokenizer.from_pretrained("babylm-seqlen/tokenizer")
        tokenizer.push_to_hub(f"babylm-seqlen/{model_type}-{seq_len}" + ("-warmup" if use_warmup else ""))

    ###
    ### Print Model Statistics
    ###

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    box_width = 70
    print("\n" + "=" * box_width)
    print(f"{'📊 MODEL TRAINING CONFIGURATION 📊':^{box_width}}")
    print("=" * box_width)
    print(f"🤖 {'Model:':<25} {model_type.upper()}")
    print(f"📏 {'Sequence Length:':<25} {seq_len}")
    print(f"🧠 {'Total parameters:':<25} {total_params}")
    print(f"🔄 {'Trainable parameters:':<25} {trainable_params}")
    print("=" * box_width + "\n")

    ###
    ### Train Model
    ###

    start_time = time.time()
    trainer.train()
    end_time = time.time()

    print(
        f"✅ Training {model_type.upper()} for seq_len {seq_len} done in {end_time - start_time:.2f}s"
    )


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type", type=str, default="opt", choices=["opt", "mamba"]
    )
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument(
        "--num_devices", type=int, default=4, help="Number of devices to use."
    )
    parser.add_argument(
        "--accumulation_steps", type=int, default=1, help="Gradient accumulation steps."
    )
    parser.add_argument("--use_deepspeed", action="store_true")
    parser.add_argument(
        "--no_push_to_hub",
        action="store_true",
        help="If set, do NOT push to the Hugging Face Hub.",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--use_warmup",
        action="store_true",
        help="If set, use learning rate warmup with sequence length scaling.",
    )

    args = parser.parse_args()

    train_model(
        model_type=args.model_type,
        seq_len=args.seq_len,
        use_deepspeed=args.use_deepspeed,
        push_to_hub=not args.no_push_to_hub,
        dry_run=args.dry_run,
        num_devices=args.num_devices,
        accumulation_steps=args.accumulation_steps,
        use_warmup=args.use_warmup,
    )

if __name__ == "__main__":
    main()