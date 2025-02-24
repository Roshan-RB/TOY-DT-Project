#!/bin/bash
# Install Rust (for dependencies that need it)
curl https://sh.rustup.rs -sSf | sh -s -- -y
export PATH="$HOME/.cargo/bin:$PATH"

# Install project dependencies
pip install --upgrade pip
pip install -r requirements.txt
