#!/usr/bin/env bash
# Train every CAD arm and seed reported in the paper (Environment A protocol: 2 GPUs, batch 256).
# Run from CHASE/NeurIPS24 after install.sh.  About 1 h per run (CHASE, fixed, vanilla) and
# 1.5 h per run (CHASE+, CHASE+B) on 2x NVIDIA T4.
set -euo pipefail
mkdir -p runs
run () {   # arm seed
  wd="runs/cad_$1_s$2"
  python main_group.py --config "cad_$1.yaml" --seed "$2" --work_dir "$wd" > "$wd.out" 2>&1
  grep -o 'best_acc: [0-9.]*%' "$wd/log.txt" | tail -1 | sed "s/^/$1 seed $2 /"
}
for s in 1 2 3;       do run vanilla "$s"; done
for s in 1 2 3 4 5;   do run chase "$s"; run fixed "$s"; run chaseplus "$s"; run chaseplusB "$s"; done
