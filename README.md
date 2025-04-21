# babylm-seqlen
How Long Can You Go? 

Train a BabyLM with Different Sequence Lengths: `--seq_len` of 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384.

```
git clone https://github.com/suchirsalhan/babylm-seqlen
python3 -m venv venvs/demo; source venvs/demo/bin/activate
bash setup.sh
pip install -e .  # From the repo root with the pyproject.toml
```

Train and push models to HuggingFace Hub
```
python train.py --model_type opt --seq_len 1024
python train.py --model_type mamba --seq_len 1024 
```
The default case `python train.py --dry_run` is 128 sequence length with OPT.

You can add  `--dry_run` and/or `--no_push_to_hub` 
```
python train.py --dry_run --no_push_to_hub
```

HPC 
```
sbatch launch_slurm.wilkes3 --model_type opt --seq_len 1024 --push_to_hub
sbatch launch_slurm.wilkes3 --model_type mamba --seq_len 1024 --push_to_hub
```

DeepSpeed Stage3 with Multiple GPU Environment for larger sequences 
```
accelerate sbatch launch_slurm.wilkes3  --multi_gpu train.py --model_type opt --seq_len 4096 --use_deepspeed
```
