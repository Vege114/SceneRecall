#!/usr/bin/env bash
set -euo pipefail
ANNOTATOR_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -f /mnt/d/SceneRecall-Setup/runtime-env.sh ]]; then
    source /mnt/d/SceneRecall-Setup/runtime-env.sh
fi
cd "$ANNOTATOR_REPO"
export PYTHONPATH="$ANNOTATOR_REPO/backend${PYTHONPATH:+:$PYTHONPATH}"
exec "$ANNOTATOR_REPO/.venv/bin/python" "$ANNOTATOR_REPO/tools/asd-annotator/server.py" \
    --library "${ANNOTATOR_REPO%/*}/SceneRecallLibrary" \
    --asset-id asset_666d87eb4e274e1fa9b7 \
    --data-dir "$ANNOTATOR_REPO/training/lrasd/manual-kaguya" \
    --port 8766 "$@"
