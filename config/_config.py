# config/_config.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class CheckpointingConfig:
    run_name: str
    save_every_n_steps: int = 500
    save_total_limit: int = 2
    output_base_dir: str = "./checkpoints"
    resume_from_checkpoint: Optional[str] = None
