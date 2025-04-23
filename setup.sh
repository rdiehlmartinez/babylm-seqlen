#!/bin/bash

# Check if the environment directory exists
if [ ! -d "env" ]; then
    # Load necessary modules
    module load python/3.11.0-icl   # Adjust this as per your system's module setup

    # Create a new virtual environment
    virtualenv -p python3.11 env
    source env/bin/activate

    # Install Git LFS (Large File Storage)
    git lfs install

    # Install the Python dependencies
    pip install --upgrade pip
    pip install -r requirements.txt

    # Install PyTorch with specific CUDA version if needed (adjust for your setup)
    pip install torch==2.5.1+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --extra-index-url https://download.pytorch.org/whl/cu118

    # Install pre-commit hooks
    pre-commit install

    # Log in to Hugging Face and WandB
    huggingface-cli login
    wandb login

else
    # If the environment already exists, just activate it
    source env/bin/activate
fi

# Source any environment variables in .env
source .env

# Update the PATH if needed
export PATH="$(pwd)/lib/bin:$PATH"

# Optional: Verify that everything is set up correctly
echo "Environment setup complete."
