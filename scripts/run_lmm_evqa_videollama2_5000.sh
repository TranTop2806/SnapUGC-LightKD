#!/usr/bin/env bash
set -euo pipefail

# Exact-architecture runner for Sun et al. LMM-EVQA VideoLLaMA2.1-7B-AV.
# Run on a cloud GPU machine after preparing data with
# scripts/prepare_lmm_evqa_videollama2_data.py.

REPO_DIR="${REPO_DIR:-/workspace/LMM-EVQA}"
REPO_URL="${REPO_URL:-https://github.com/sunwei925/LMM-EVQA.git}"
REPO_COMMIT="${REPO_COMMIT:-b3434ee576ad42d5141be8d6c5c45734a9313794}"
WORK_DIR="${REPO_DIR}/VideoLLaMA2-audio_visual"

DATA_DIR="${DATA_DIR:-/workspace/snapugc_lmm_evqa_5000}"
TRAIN_JSON="${TRAIN_JSON:-${DATA_DIR}/train.json}"
VAL_JSON="${VAL_JSON:-${DATA_DIR}/val.json}"
ALL_JSON="${ALL_JSON:-${DATA_DIR}/all.json}"
MODEL_ROOT="${MODEL_ROOT:-/workspace/videollama2weights}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/videollama2_evqa_snapugc_5000_mse}"
RUN_NAME="${RUN_NAME:-snapugc_5000_lmm_evqa_videollama2_exact}"

if [[ ! -d "${REPO_DIR}/.git" ]]; then
  git clone "${REPO_URL}" "${REPO_DIR}"
fi
git -C "${REPO_DIR}" fetch --all --tags
git -C "${REPO_DIR}" checkout "${REPO_COMMIT}"

cd "${WORK_DIR}"

if [[ ! -f "${MODEL_ROOT}/audio_tower.bin" || ! -f "${MODEL_ROOT}/mm_projector_a.bin" ]]; then
  echo "Missing VideoLLaMA2 weights under ${MODEL_ROOT}."
  echo "Run the official download step in ${WORK_DIR}: python download_model_weight.py"
  exit 2
fi

if [[ ! -f "${TRAIN_JSON}" || ! -f "${VAL_JSON}" ]]; then
  echo "Missing train/val JSON. Expected:"
  echo "  ${TRAIN_JSON}"
  echo "  ${VAL_JSON}"
  exit 2
fi

python -u videollama2/train_EVQA.py \
  --model_type videollama2_qwen2 \
  --model_path "${MODEL_ROOT}" \
  --data_folder datasets \
  --data_path "${TRAIN_JSON}" \
  --vision_tower google/siglip-so400m-patch14-384 \
  --audio_tower "${MODEL_ROOT}/audio_tower.bin" \
  --pretrain_mm_mlp_adapter_a "${MODEL_ROOT}/mm_projector_a.bin" \
  --mm_projector_type stc_connector_v35 \
  --mm_projector_a_type mlp2x_gelu \
  --va True \
  --tune_audio_tower True \
  --tune_adapter_llm True \
  --tune_mm_mlp_adapter_a True \
  --mm_vision_select_layer -2 \
  --num_frames 8 \
  --bf16 True \
  --tf32 True \
  --fp16 False \
  --loss_type mse \
  --output_dir "${OUTPUT_DIR}" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 6 \
  --gradient_accumulation_steps 4 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 17 \
  --save_total_limit 1 \
  --learning_rate 5e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --model_max_length 2048 \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --lazy_preprocess True \
  --report_to none \
  --run_name "${RUN_NAME}" | tee "${OUTPUT_DIR}.training_log.txt"

# Official test.py reads ./val.json and writes ./submission.csv.
ln -sf "${VAL_JSON}" "${WORK_DIR}/val.json"
python -u videollama2/test.py \
  --model-path "${OUTPUT_DIR}" \
  --modal-type av | tee "${OUTPUT_DIR}.validation_log.txt"

mkdir -p "${OUTPUT_DIR}/eval"
cp "${WORK_DIR}/submission.csv" "${OUTPUT_DIR}/eval/val_submission.csv"
cp "${OUTPUT_DIR}.training_log.txt" "${OUTPUT_DIR}/eval/training_log.txt"
cp "${OUTPUT_DIR}.validation_log.txt" "${OUTPUT_DIR}/eval/validation_log.txt"
echo "Saved validation predictions to ${OUTPUT_DIR}/eval/val_submission.csv"

if [[ -f "${ALL_JSON}" ]]; then
  ln -sf "${ALL_JSON}" "${WORK_DIR}/val.json"
  python -u videollama2/test.py \
    --model-path "${OUTPUT_DIR}" \
    --modal-type av | tee "${OUTPUT_DIR}.all_log.txt"
  cp "${WORK_DIR}/submission.csv" "${OUTPUT_DIR}/eval/all_submission.csv"
  cp "${OUTPUT_DIR}.all_log.txt" "${OUTPUT_DIR}/eval/all_log.txt"
  echo "Saved all-split teacher predictions to ${OUTPUT_DIR}/eval/all_submission.csv"
fi
