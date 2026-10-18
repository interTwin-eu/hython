#!/usr/bin/env bash
# Run the dPL multicycle in the `emulator` env, whatever env the calling shell has.
# The env's bin goes first on PATH because run_dpl_cycle.py starts `itwinai` through a shell.
set -euo pipefail
cd "$(dirname "$0")"

ENV_BIN=/mnt/CEPH_PROJECTS/A_DROP/pilot5/dev/conda/emulator/bin
export PATH="$ENV_BIN:$PATH"

nohup "$ENV_BIN/python" run_dpl_cycle.py >> dpl_production.log 2>&1 &
echo "started PID $!, log: $(pwd)/dpl_production.log"
