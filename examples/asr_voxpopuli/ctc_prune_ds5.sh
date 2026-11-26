#!/bin/bash
# CTC-guided pruning at ds=5, one model per threshold (the fixed-threshold "CTC" curve in the paper).
# For each CTC-blank threshold: train projector + LoRA on VoxPopuli-EN with that threshold,
# decode the test set with the same threshold, and score WER.
#
# A frame is dropped when its argmax is the CTC blank AND its blank probability is >= the threshold,
# so 1.1 keeps every frame and 0.0 keeps only non-blank frames.
#
# Overridable via environment: ADACTC_ROOT, DATA_DIR, LLM_PATH, ENCODER_PATH, JSONL_DIR, OUT_DIR,
# THRESHOLDS (space-separated subset of the list below), CUDA_VISIBLE_DEVICES.
set -euo pipefail

export ADACTC_ROOT="${ADACTC_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
export DATA_DIR="${DATA_DIR:-$ADACTC_ROOT/data}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
USER="${USER:-$(whoami)}"
export PYTHONPATH="$ADACTC_ROOT/src"
cd "$ADACTC_ROOT"

task=asr
code_dir=examples/asr_voxpopuli
llm_path="${LLM_PATH:-$ADACTC_ROOT/models/eurollm-1.7b}"
encoder_path="${ENCODER_PATH:-$ADACTC_ROOT/models/whisper_ctc_partial_unfreeze.pth}"
jsonl_dir="${JSONL_DIR:-$DATA_DIR/voxpopuli_jsonl}"   # train.jsonl, val.jsonl, test.jsonl
out_root="${OUT_DIR:-$ADACTC_ROOT/exp/ctc_prune_ds5_asr_voxpopuli}"
num_epochs=4

# Threshold -> token rate (tokens/s) on VoxPopuli-EN test, with the partially fine-tuned CTC head:
#   1.1 -> 10 | 0.9999951124191284 -> 8.34 | 0.999987006187439 -> 7.14 | 0.9999709129333496 -> 6.25
#   0.9999358952045441 -> 5.56 | 0.9998627007007599 -> 5 | 0.9931716918945312 -> 3.34
#   0.89990234375 -> 2.5 | 0.0 -> 1.72 (all blank-argmax frames removed)
default_thresholds="1.1 0.9999951124191284 0.999987006187439 0.9999709129333496 0.9999358952045441 0.9998627007007599 0.9931716918945312 0.89990234375 0.0"
read -r -a thresholds <<< "${THRESHOLDS:-$default_thresholds}"

for f in "$llm_path" "$encoder_path" "$jsonl_dir/train.jsonl" "$jsonl_dir/val.jsonl" "$jsonl_dir/test.jsonl"; do
    [[ -e "$f" ]] || { echo "Missing: $f" >&2; exit 1; }
done

# Model / pruning settings shared by training and inference.
model_args=(
    ++model_config.llm_name=eurollm-1.7b
    ++model_config.llm_path="$llm_path"
    ++model_config.llm_dim=2048
    ++model_config.encoder_name=whisper-ctc
    ++model_config.encoder_path="$encoder_path"
    ++model_config.encoder_dim=1280
    ++model_config.encoder_pad=0
    ++model_config.encoder_ctc_layer=32
    ++model_config.ctc_vocab_size=-1
    ++model_config.encoder_projector=pruning
    ++model_config.encoder_projector_ds_rate=5
    ++model_config.downsampling_type=linear
    ++model_config.pruning_type=ctc
    ++model_config.normalize=true
    ++dataset_config.dataset=speech_dataset
    ++dataset_config.input_type=mel
    ++dataset_config.mel_size=128
    ++train_config.model_name=$task
    ++train_config.freeze_encoder=true
    ++train_config.freeze_llm=true
    ++train_config.use_peft=true
    ++train_config.batching_strategy=custom
)

for thr in "${thresholds[@]}"; do
    out_dir="$out_root/thr${thr}"
    mkdir -p "$out_dir"
    echo ">> threshold $thr -> $out_dir"

    # 1. Training: frozen Whisper-CTC encoder and LLM; projector + LoRA are trained.
    torchrun --nnodes 1 --nproc_per_node 1 "$code_dir/finetune_${task}.py" \
        --config-path "conf" --config-name "prompt.yaml" \
        "${model_args[@]}" \
        ++model_config.encoder_prune_thr="$thr" \
        ++train_config.enable_fsdp=false ++train_config.enable_ddp=true \
        ++train_config.use_fp16=true \
        ++train_config.peft_config.peft_method=lora \
        ++train_config.warmup_steps=1000 \
        ++train_config.total_steps=1000000 \
        ++train_config.validation_interval=5000 \
        ++train_config.batch_size_training=4 \
        ++train_config.val_batch_size=4 \
        ++train_config.num_workers_dataloader=4 \
        ++train_config.lr=1e-4 \
        ++train_config.num_epochs=$num_epochs \
        ++dataset_config.train_data_path="$jsonl_dir/train.jsonl" \
        ++dataset_config.val_data_path="$jsonl_dir/val.jsonl" \
        ++metric=acc \
        ++train_config.output_dir="$out_dir" \
        ++hydra.run.dir="$out_dir" \
        ++log_config.use_wandb=false \
        > "$out_dir/train.log" 2>&1

    # Dataloader workers occasionally outlive the trainer; stop only this recipe's leftovers.
    pkill -9 -u "$USER" -f "$code_dir/finetune_${task}.py" || true

    # 2. Latest checkpoint written by the trainer.
    ckpt_dir=$(ls -d "$out_dir"/${task}_epoch_*_step_* 2>/dev/null | sort -V | tail -n 1 || true)
    if [[ ! -d "$ckpt_dir" ]]; then
        echo "No checkpoint found in $out_dir (see $out_dir/train.log)" >&2
        continue
    fi
    decode_log="$out_dir/decode_test"

    # 3. Inference with the same threshold.
    python -u "$code_dir/inference_${task}_batch.py" \
        --config-path "conf" --config-name "prompt.yaml" \
        hydra.run.dir="$out_dir" \
        "${model_args[@]}" \
        ++model_config.encoder_prune_thr="$thr" \
        ++dataset_config.val_data_path="$jsonl_dir/test.jsonl" \
        ++dataset_config.inference_mode=true \
        ++train_config.num_epochs=1 \
        ++train_config.val_batch_size=1 \
        ++decode_log="$decode_log" \
        ++ckpt_path="$ckpt_dir/model.pt" \
        > "$out_dir/inference.log" 2>&1

    # 4. Scoring.
    python src/slam_llm/utils/compute_wer.py "${decode_log}_gt" "${decode_log}_pred" "$out_dir/WER.txt"
done
