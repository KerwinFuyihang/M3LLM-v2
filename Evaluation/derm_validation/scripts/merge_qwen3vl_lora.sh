#!/usr/bin/env bash
set -euo pipefail
MODEL=qwen3vl exec "$(dirname "$0")/merge_derm_lora.sh"
