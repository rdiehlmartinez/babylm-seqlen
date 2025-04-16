import os
import torch

def save_checkpoint(model, optimizer, scheduler, output_dir, step):
    os.makedirs(output_dir, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "step": step
    }
    path = os.path.join(output_dir, f"checkpoint-{step}.pt")
    torch.save(checkpoint, path)
    print(f"✅ Saved checkpoint to {path}")

def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    print(f"✅ Loaded checkpoint from {checkpoint_path}")
    return checkpoint.get("step", 0)
