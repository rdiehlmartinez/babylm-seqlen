import os
import time
import shutil
from pathlib import Path

import torch

from datasets import load_dataset
from transformers import (
    OPTConfig,
    OPTForCausalLM,
    Trainer,
    TrainingArguments,
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

    Ok but also this is not entirely correct, because we actually compute the number of tokens 
    rather than the number of words.
    """
    def __init__(self, total_steps, seq_len):
        super().__init__()

        self.seq_len = seq_len
        total_tokens = total_steps * GLOBAL_BATCH_SIZE * seq_len
        self.token_to_word_ratio = total_tokens/1_000_000_000

        self.checkpoint_tokens = (
            [int(self.token_to_word_ratio * i * 1_000_000) for i in range(1, 11)] +      # 1M to 10M
            [int(self.token_to_word_ratio * i * 10_000_000) for i in range(2, 11)] +     # 20M to 100M
            [int(self.token_to_word_ratio * i * 100_000_000) for i in range(2, 11)]      # 200M to 1B
        )
        
        self.next_checkpoint_idx = 0

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        tokens_seen = state.global_step * GLOBAL_BATCH_SIZE * self.seq_len
        if (self.next_checkpoint_idx < len(self.checkpoint_tokens) and
            tokens_seen >= self.checkpoint_tokens[self.next_checkpoint_idx]):
            print(f"\nDEBUG: Triggering checkpoint at step {state.global_step}")
            control.should_save = True
            words_seen = int(self.checkpoint_tokens[self.next_checkpoint_idx]/self.token_to_word_ratio)
            debug_message = (
                f"Checkpoint at {words_seen:,} words ({self.checkpoint_tokens[self.next_checkpoint_idx]:,} tokens) "
                f"| Step {state.global_step:,} | Progress: {self.next_checkpoint_idx + 1}/{len(self.checkpoint_tokens)}"
            )
            print(f"DEBUG: {debug_message}")
            self.next_checkpoint_idx += 1
        return control

import json 

from transformers.trainer_utils import (
    HubStrategy,
    PREFIX_CHECKPOINT_DIR,
)

from transformers.utils import (
    CONFIG_NAME,
    WEIGHTS_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
)

TRAINING_ARGS_NAME = "training_args.bin"

from huggingface_hub import upload_folder

class CustomTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        self.seq_len = kwargs.pop("seq_len")
        self.total_steps = kwargs.pop("total_steps")

        total_tokens = self.total_steps * GLOBAL_BATCH_SIZE * self.seq_len

        self.token_to_word_ratio = total_tokens/1_000_000_000

        # NOTE: Literally repeating this from the Checkpointing class but honestly sue me 
        # I don't care anymore 
        self.checkpoint_words = (
            [int(i * 1_000_000) for i in range(1, 11)] +      # 1M to 10M
            [int(i * 10_000_000) for i in range(2, 11)] +     # 20M to 100M
            [int(i * 100_000_000) for i in range(2, 11)]      # 200M to 1B
        )

        self.next_checkpoint_idx = 0

        super().__init__(*args, **kwargs)

    def _push_from_checkpoint(self, checkpoint_folder):
        # Only push from one node.
        if not self.is_world_process_zero() or self.args.hub_strategy == HubStrategy.END:
            return
        # If we haven't finished the last push, we don't do this one unless args.hub_always_push=True.
        if not self.args.hub_always_push and self.push_in_progress is not None and not self.push_in_progress.is_done():
            return

        output_dir = self.args.output_dir
        # To avoid a new synchronization of all model weights, we just copy the file from the checkpoint folder
        modeling_files = [CONFIG_NAME, WEIGHTS_NAME, SAFE_WEIGHTS_NAME]
        #  Add sharded checkpoints if we have an index
        for index_file in [WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_INDEX_NAME]:
            index_path = os.path.join(checkpoint_folder, index_file)
            if os.path.isfile(index_path):
                modeling_files.append(index_file)
                with open(index_path) as f:
                    index = json.loads(f.read())
                shard_files = list(set(index["weight_map"].values()))
                modeling_files.extend(shard_files)

        for modeling_file in modeling_files:
            if os.path.isfile(os.path.join(checkpoint_folder, modeling_file)):
                shutil.copy(os.path.join(checkpoint_folder, modeling_file), os.path.join(output_dir, modeling_file))
        # Saving the processing class is fast and we don't know how many files it may have spawned, so we resave it to be sure.
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        # Same for the training arguments
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

        commit_message = f"Checkpoint step: {self.state.global_step:,} | Target words: {self.checkpoint_words[self.next_checkpoint_idx]:,} | Actual tokens: {self.seq_len*self.state.global_step*GLOBAL_BATCH_SIZE:,} | Actual words: {self.state.global_step*GLOBAL_BATCH_SIZE*self.seq_len/self.token_to_word_ratio:,} | Progress: {self.next_checkpoint_idx + 1}/{len(self.checkpoint_words)}"
        self.next_checkpoint_idx += 1

        max_retries = 5
        for attempt in range(max_retries):
            try:
                _ = upload_folder(
                    repo_id=self.hub_model_id,
                    folder_path=output_dir,
                    commit_message=commit_message,
                    token=self.args.hub_token,
                    run_as_future=False,
                    ignore_patterns=["_*", f"{PREFIX_CHECKPOINT_DIR}-*"],
                )
                break  # Success!
            except Exception as e:
                print(f"Upload attempt {attempt+1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(15)  # Wait before retrying
                else:
                    print("Max upload retries reached. Skipping this checkpoint push.")

        if self.args.hub_strategy in [HubStrategy.CHECKPOINT, HubStrategy.ALL_CHECKPOINTS]:
            path_in_repo = (
                "last-checkpoint" if self.args.hub_strategy == HubStrategy.CHECKPOINT else Path(checkpoint_folder).name
            )
            _ = upload_folder(
                repo_id=self.hub_model_id,
                folder_path=checkpoint_folder,
                path_in_repo=path_in_repo,
                commit_message=commit_message + ", checkpoint",
                token=self.args.hub_token,
                run_as_future=False,
            )


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
    special_id="",
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

    suffix = ("-warmup" if use_warmup else "") + (f"-{special_id}" if special_id else "")

    if dry_run:
        train_dataset = train_dataset.select(range(100))
        output_dir = f"./dryruns/{model_type}-babylm-{seq_len}{suffix}"
    else:
        output_dir = f"./checkpoints/{model_type}-babylm-{seq_len}{suffix}"

    os.makedirs(output_dir, exist_ok=True)

    run_name = f"{model_type}_babylm_{seq_len}{suffix}"

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

    # Initial checkpointing rate is every 1M words, which is 1% of an epoch.
    # We then increase the checkpointing rate by a factor of 10 every 10 checkpoints.
    total_steps = TRAIN_EPOCHS * len(train_dataset) // GLOBAL_BATCH_SIZE
    warmup_steps = int(total_steps * 0.05) if use_warmup else 0  # 5% of total steps for warmup

    custom_checkpointing_callback = CustomCheckpointingCallback(total_steps, seq_len)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=accumulation_steps,
        num_train_epochs=TRAIN_EPOCHS,
        eval_strategy="no",
        save_strategy="no",
        bf16=True,
        report_to="wandb",
        run_name=run_name,
        deepspeed=get_deepspeed_config(accumulation_steps, num_devices) if use_deepspeed else None,
        logging_steps=max(total_steps // 1000, 1),
        disable_tqdm=False,
        push_to_hub=push_to_hub,
        hub_model_id=f"babylm-seqlen/{model_type}-{seq_len}{suffix}",
        hub_strategy="every_save",
        learning_rate=5e-5*(seq_len/64) if use_warmup else 5e-5,
        warmup_steps=warmup_steps,
        lr_scheduler_type="linear" if use_warmup else "constant",
    )

    print(f"Training arguments:\n{training_args}")

    ###
    ### Setup Trainer
    ###

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        callbacks=[custom_checkpointing_callback],
        total_steps=total_steps,
        seq_len=seq_len,
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
    parser.add_argument(
        "--use_warmup",
        action="store_true",
        help="If set, use learning rate warmup with sequence length scaling.",
    )
    parser.add_argument(
        "--special_id",
        type=str,
        help="Special ID to append to the model name and project name.",
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
        special_id=args.special_id,
    )

if __name__ == "__main__":
    main()