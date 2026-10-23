#!/usr/bin/env bash
# Prepare a CHASE checkout with everything needed to reproduce the paper.
#   bash install.sh [TARGET_DIR]          (default: ./CHASE)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
TARGET="${1:-$HERE/CHASE}"
COMMIT=0a62973af96054f24f650b21c22d974c1d482387     # Necolizer/CHASE commit used for every experiment

if [ ! -d "$TARGET" ]; then
  git clone https://github.com/Necolizer/CHASE.git "$TARGET"
fi
git -C "$TARGET" fetch -q --depth 1 origin "$COMMIT" 2>/dev/null || true
git -C "$TARGET" checkout -q "$COMMIT"
REPO="$TARGET/NeurIPS24"

# 1. wrappers (CHASE+, CHASE+B, fixed origin, vanilla)
cp "$HERE"/src/chase_plus.py "$HERE"/src/chase_plus_repo.py "$HERE"/src/fixed_origin.py "$REPO/model/Wrapper/"
for m in fixed_origin chase_plus chase_plus_repo; do
  grep -q "import $m" "$REPO/model/Wrapper/__init__.py" || printf '\nfrom . import %s\n' "$m" >> "$REPO/model/Wrapper/__init__.py"
done

# 2. loss LSCE_AUX_GROUP (classification + auxiliary loss returned by CHASEPlusRepo)
if ! grep -q LSCE_AUX_GROUP "$REPO/utils/loss.py"; then
  git -C "$TARGET" apply "$HERE/patches/loss_LSCE_AUX_GROUP.patch"
fi

# 3. probes, eps calibration and configs
cp "$HERE"/probes/*.py "$HERE/scripts/calibrate_eps.py" "$REPO/"
cp "$HERE"/configs/*.yaml "$REPO/"

echo "Ready: $REPO"
echo "Put the CAD data in $REPO/data/collective (or edit dataset_dir in the cad_*.yaml files)."
