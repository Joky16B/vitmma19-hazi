

set -euo pipefail

echo "[run.sh] Starting full pipeline run at $(date --iso-8601=seconds)"

python 01-data-preprocessing.py
python 02-training.py
python 03-evaluation.py
python 04-inference.py

echo "[run.sh] Pipeline finished at $(date --iso-8601=seconds)"
