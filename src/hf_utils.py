# src/hf_utils.py

from huggingface_hub import HfApi

def save_to_hf(model_type, output_dir, repo_id):
    api = HfApi()
    api.create_repo(repo_id, exist_ok=True)
    api.upload_folder(folder_path=output_dir, repo_id=repo_id)

"""
from huggingface_hub import HfApi, create_repo, upload_folder

def save_to_hf(model_type, local_path, checkpointing_config):
    repo_id = checkpointing_config.hf_checkpoint.repo_id or f"babylm-seqlen/{model_type}-trained"
    create_repo(repo_id, exist_ok=True)
    upload_folder(
        repo_id=repo_id,
        folder_path=local_path,
        commit_message=f"Upload model: {model_type}",
        repo_type="model",
    )
    print(f"✅ Uploaded to HF Hub: https://huggingface.co/{repo_id}")
"""
