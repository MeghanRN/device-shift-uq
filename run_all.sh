#!/bin/bash

DATASETS=("dcase" "cwru" "sonyc")
MODELS=("cpmobile" "dynacp" "grucnn")
SEEDS=(0 1 2)

for DATASET in "${DATASETS[@]}"
do
  for MODEL in "${MODELS[@]}"
  do
    for SEED in "${SEEDS[@]}"
    do
      RESULT_DIR="results/${DATASET}/${MODEL}/seed_${SEED}"
      EVAL_FILE="${RESULT_DIR}/eval_uq_summary.json"

      if [ -f "$EVAL_FILE" ]; then
        echo "Skipping ${DATASET} ${MODEL} seed=${SEED} (already done)"
        continue
      fi

      echo "Running ${DATASET} ${MODEL} seed=${SEED}"

      python -m training.train \
        --dataset "$DATASET" \
        --model "$MODEL" \
        --seed "$SEED"

      python -m training.evaluate \
        --dataset "$DATASET" \
        --model "$MODEL" \
        --seed "$SEED"

    done
  done
done

echo "All experiments complete"