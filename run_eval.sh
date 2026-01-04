#!/bin/bash
cd /opt/imagry/DiffDynamics

echo "Starting evaluation at $(date)" > evaluation/eval_status.txt
echo "PID: $$" >> evaluation/eval_status.txt

python scripts/evaluate_diffusion.py \
  --model-path training/diffusion_prior/best.pt \
  --output-dir evaluation/final_with_smoothing \
  --n-test-maps 5 \
  --obs-counts 10 20 50 100 200 500 \
  --scheduler DDPM \
  --num-inference-steps 50 \
  --guidance-scale 1.0 \
  --gradient-smoothing-sigma 2.0 \
  --seed 42 >> evaluation/final_smoothing_eval.log 2>&1

echo "Completed at $(date)" >> evaluation/eval_status.txt
echo "Exit code: $?" >> evaluation/eval_status.txt

