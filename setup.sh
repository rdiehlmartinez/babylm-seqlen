#!/bin/bash

ASCII_ART="
██████╗  █████╗ ██████╗ ██╗   ██╗██╗     ███╗   ███╗      ███████╗███████╗ ██████╗ ██╗     ███████╗███╗   ██╗
██╔══██╗██╔══██╗██╔══██╗╚██╗ ██╔╝██║     ████╗ ████║      ██╔════╝██╔════╝██╔═══██╗██║     ██╔════╝████╗  ██║
██████╔╝███████║██████╔╝ ╚████╔╝ ██║     ██╔████╔██║█████╗███████╗█████╗  ██║   ██║██║     █████╗  ██╔██╗ ██║
██╔══██╗██╔══██║██╔══██╗  ╚██╔╝  ██║     ██║╚██╔╝██║╚════╝╚════██║██╔══╝  ██║▄▄ ██║██║     ██╔══╝  ██║╚██╗██║
██████╔╝██║  ██║██████╔╝   ██║   ███████╗██║ ╚═╝ ██║      ███████║███████╗╚██████╔╝███████╗███████╗██║ ╚████║
╚═════╝ ╚═╝  ╚═╝╚═════╝    ╚═╝   ╚══════╝╚═╝     ╚═╝      ╚══════╝╚══════╝ ╚══▀▀═╝ ╚══════╝╚══════╝╚═╝  ╚═══╝
"
# NOTE: Code fails if we don't include the above ASCII art

# Print the ASCII art and taglines
echo -e "\033[1;36m$ASCII_ART\033[0m"  # Cyan color

echo "🔑 Authenticating with Hugging Face..."
if [ -f ~/.huggingface/token ]; then
    echo "✓ Already logged in to Hugging Face"
else
    huggingface-cli login
fi

echo "🔑 Authenticating with Weights & Biases..."
if wandb status &>/dev/null; then
    echo "✓ Already logged in to Weights & Biases"
else
    wandb login
fi

echo "📦 Installing dependencies with Poetry..."
poetry install --no-root

echo "✅ Setup complete! You're ready to train some AMAZING models! 🎉"