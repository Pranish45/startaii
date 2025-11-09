#!/usr/bin/env bash
set -euo pipefail

# Install system prerequisites (may vary by build image)
apt-get update && apt-get install -y curl build-essential

# Install rustup (non-interactive)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Ensure cargo in PATH for the rest of the build
export PATH="$HOME/.cargo/bin:$PATH"

# Now run the default steps: install pip deps and continue with your existing build commands
pip install --upgrade pip
pip install -r requirements.txt

# run remaining build steps (example)
# python manage.py collectstatic --noinput
# npm run build
