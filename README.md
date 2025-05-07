# babylm-seqlen
How Long Can You Go? 

Train a BabyLM with Different Sequence Lengths: `--seq_len` of 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384.

```
git clone https://github.com/rdiehlmartinez/babylm-seqlen
bash setup.sh
pip install -e .  # From the repo root with the pyproject.toml
```

Train and push models to HuggingFace Hub
```
poetry run train --model_type opt --seq_len 1024
poetry run train --model_type mamba --seq_len 1024 
```
The default case `poetry run train --dry_run` is 128 sequence length with OPT.

You can add  `--dry_run` and/or `--no_push_to_hub` 
```
poetry run train --dry_run --no_push_to_hub
```

HPC 
```
sbatch launch_slurm.wilkes3 --model_type opt --seq_len 1024 
sbatch launch_slurm.wilkes3 --model_type mamba --seq_len 1024 
```
