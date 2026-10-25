#!/bin/bash
# Ablation: separate the effect of width from the cosine schedule + patience 50.
cd "$(dirname "$0")/../.."
for seed in 42 43; do
  d=runs/ablation/small_cosine_p50/seed_$seed; mkdir -p $d
  python train.py --hidden 128 64 32 16 --scheduler cosine --patience 50 --seed $seed --threads 1 --out $d > $d/train.log 2>&1 &
  d=runs/ablation/wide_plateau_p20/seed_$seed; mkdir -p $d
  python train.py --hidden 256 128 64 32 16 --scheduler plateau --patience 20 --seed $seed --threads 1 --out $d > $d/train.log 2>&1 &
done
wait
grep -H '"test_mse"' runs/ablation/*/seed_*/metrics.json
