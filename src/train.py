import os
import time

from datasets import load_dataset
from transformers import (
    OPTConfig,
    OPTForCausalLM,
    Trainer,
    TrainingArguments,
)

from transformers.trainer_utils import (
    SaveStrategy,
    IntervalStrategy,
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
    A [`TrainerCallback`] that adjusts the rate of checkpointing according to the BabyLM specifications:
        Every 10 checkpoints, we increase the checkpointing rate by a factor of 10.  
    """

    def __init__(self, total_steps):
        super().__init__()
        self.num_checkpoints = 0
        self.rate = 0.001
        self.total_steps = total_steps

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if (
            args.save_strategy == SaveStrategy.STEPS
            and state.save_steps > 0
            and state.global_step % state.save_steps == 0
        ):
            control.should_save = True
            self.num_checkpoints += 1
            # Handle very small initial save_steps due to very large
            # sequence lengths by recalculating save_steps after each checkpoint.
            segment_size = self.rate * self.total_steps
            next_save_step = int((self.num_checkpoints+1) * segment_size)
            state.save_steps = next_save_step if state.global_step - next_save_step > 0 else next_save_step+1
            print(f"Checkpointing at step {state.global_step}")
            if self.num_checkpoints == 9:
                self.rate *= 10
                self.num_checkpoints = 0

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
        output_dir = f"./dryruns/{model_type}-babylm-{seq_len}"
    else:
        output_dir = f"./checkpoints/{model_type}-babylm-{seq_len}"

    os.makedirs(output_dir, exist_ok=True)

    run_name = f"{model_type}_babylm_{seq_len}"

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
    custom_checkpointing_callback = CustomCheckpointingCallback(total_steps)
    print(f'Initial save steps set to 1% of an epoch: {initial_save_steps:.2f} steps')

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
        hub_model_id=f"babylm-seqlen/{model_type}-{seq_len}",
        hub_strategy="every_save",
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

    args = parser.parse_args()

    train_model(
        model_type=args.model_type,
        seq_len=args.seq_len,
        use_deepspeed=args.use_deepspeed,
        push_to_hub=not args.no_push_to_hub,
        dry_run=args.dry_run,
        num_devices=args.num_devices,
        accumulation_steps=args.accumulation_steps,
    )

if __name__ == "__main__":
    main()