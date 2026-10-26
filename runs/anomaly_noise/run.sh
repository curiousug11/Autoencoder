#!/bin/bash
# Train denoising autoencoders: one per code and noise level (white noise per bin, ppm),
# same architecture/schedule/split as runs/anomaly. 4 runs in parallel, one thread each.
cd "$(dirname "$0")/../.."
for sigma in 10 20 30 50; do
  for code in pos prt taurex; do
    echo "$sigma $code"
  done
done | xargs -P 4 -n 2 sh -c '
  d=runs/anomaly_noise/sigma$0/train_$1; mkdir -p $d
  python train.py --dataset $1 --cut-datasets pos prt taurex --seed 42 --split-seed 42 \
    --hidden 256 128 64 32 16 --latent 8 --scheduler cosine --patience 50 \
    --noise-ppm $0 --threads 1 --out $d > $d/train.log 2>&1 && echo "sigma $0 $1 done" || echo "sigma $0 $1 FAILED"'
