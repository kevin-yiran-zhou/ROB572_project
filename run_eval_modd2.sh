#!/usr/bin/env bash
# Run evaluation on MODD2 rectified video sequences.
# Usage: bash run_eval_modd2.sh
# Left camera only (*L.jpg), matching combined.py IMAGE_GLOB_PATTERN.
#
# Prerequisites:
#   - MODD2 data extracted under ./MODD2_video_data_rectified/
#   - Model weights in place (segmentation/model/segformer_instance_aware_best.pth,
#     depth/model/small/)
#   - Python environment with torch, cv2, numpy, PIL installed

set -e

MODD2_ROOT="MODD2_video_data_rectified/video_data"

# --- Sequence 1: kope67 ---
SEQ1="${MODD2_ROOT}/kope67-00-00025200-00025670/framesRectified"
OUT1="eval_results_kope67"

echo "=== Evaluating sequence 1: kope67 ==="
python eval_detection.py \
    --image-dir "$SEQ1" \
    --image-glob '*L.jpg' \
    --out-dir "$OUT1"

echo "=== Generating plots for kope67 ==="
python plot_eval.py --eval-dir "$OUT1" --save

# --- Sequence 2: kope71 ---
SEQ2="${MODD2_ROOT}/kope71-01-00011210-00011320/framesRectified"
OUT2="eval_results_kope71"

echo "=== Evaluating sequence 2: kope71 ==="
python eval_detection.py \
    --image-dir "$SEQ2" \
    --image-glob '*L.jpg' \
    --out-dir "$OUT2"

echo "=== Generating plots for kope71 ==="
python plot_eval.py --eval-dir "$OUT2" --save

# --- Combined plots across both sequences ---
echo "=== Generating combined plots ==="
python plot_eval.py --eval-dir "${OUT1},${OUT2}" --save

echo "=== Done ==="
echo "Results saved to: ${OUT1}/ and ${OUT2}/"
