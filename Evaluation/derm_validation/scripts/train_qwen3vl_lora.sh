#!/usr/bin/env bash
set -euo pipefail
MODEL=qwen3vl exec "$(dirname "$0")/train_derm_lora.sh"
