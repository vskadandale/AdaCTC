#!/bin/bash
# Train the Adaptive (AdaCTC) ASR model on VoxPopuli-EN.
#
# One model for all token rates: during training the CTC-blank threshold is drawn at random from
# the set below for every utterance, so the projector + LoRA learn to handle any pruning level.
# Decode the result at a chosen rate with decode_adactc.sh.
#
# Overridable via environment: ADACTC_ROOT, DATA_DIR, LLM_PATH, ENCODER_PATH, JSONL_DIR, OUT_DIR,
# THRESHOLDS (training threshold set), CUDA_VISIBLE_DEVICES.
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
jsonl_dir="${JSONL_DIR:-$DATA_DIR/voxpopuli_jsonl}"   # train.jsonl, val.jsonl
out_dir="${OUT_DIR:-$ADACTC_ROOT/exp/train_adactc_asr_voxpopuli}"
num_epochs=10

# Thresholds sampled during training (token rates 10 ... 1.72 TPS; see decode_adactc.sh for the mapping).
default_thresholds="0.0 0.89990234375 0.9931716918945312 0.9998627007007599 0.9999358952045441 0.9999709129333496 0.999987006187439 0.9999951124191284 1.1"
read -r -a thresholds <<< "${THRESHOLDS:-$default_thresholds}"
threshold_list="[$(IFS=,; echo "${thresholds[*]}")]"

for f in "$llm_path" "$encoder_path" "$jsonl_dir/train.jsonl" "$jsonl_dir/val.jsonl"; do
    [[ -e "$f" ]] || { echo "Missing: $f" >&2; exit 1; }
done
mkdir -p "$out_dir"
echo ">> training AdaCTC ($task) with thresholds $threshold_list -> $out_dir"

# Frozen Whisper-CTC encoder and LLM; projector + LoRA are trained.
torchrun --nnodes 1 --nproc_per_node 1 "$code_dir/finetune_${task}.py" \
    --config-path "conf" --config-name "prompt.yaml" \
    ++model_config.llm_name=eurollm-1.7b \
    ++model_config.llm_path="$llm_path" \
    ++model_config.llm_dim=2048 \
    ++model_config.encoder_name=whisper-ctc \
    ++model_config.encoder_path="$encoder_path" \
    ++model_config.encoder_dim=1280 \
    ++model_config.encoder_pad=0 \
    ++model_config.encoder_ctc_layer=32 \
    ++model_config.ctc_vocab_size=-1 \
    ++model_config.encoder_projector=pruning \
    ++model_config.encoder_projector_ds_rate=5 \
    ++model_config.downsampling_type=linear \
    ++model_config.pruning_type=ctc \
    ++model_config.use_dynamic_pruning=true \
    ++model_config.dynamic_prune_thresholds_set="$threshold_list" \
    ++model_config.normalize=true \
    ++dataset_config.dataset=speech_dataset \
    ++dataset_config.input_type=mel \
    ++dataset_config.mel_size=128 \
    ++dataset_config.train_data_path="$jsonl_dir/train.jsonl" \
    ++dataset_config.val_data_path="$jsonl_dir/val.jsonl" \
    ++train_config.model_name=$task \
    ++train_config.freeze_encoder=true \
    ++train_config.freeze_llm=true \
    ++train_config.use_peft=true \
    ++train_config.peft_config.peft_method=lora \
    ++train_config.batching_strategy=custom \
    ++train_config.enable_fsdp=false ++train_config.enable_ddp=true \
    ++train_config.use_fp16=true \
    ++train_config.warmup_steps=1000 \
    ++train_config.total_steps=1000000 \
    ++train_config.validation_interval=5000 \
    ++train_config.batch_size_training=4 \
    ++train_config.val_batch_size=4 \
    ++train_config.num_workers_dataloader=4 \
    ++train_config.lr=1e-4 \
    ++train_config.num_epochs=$num_epochs \
    ++metric=acc \
    ++train_config.output_dir="$out_dir" \
    ++hydra.run.dir="$out_dir" \
    ++log_config.use_wandb=false \
    > "$out_dir/train.log" 2>&1

# Dataloader workers occasionally outlive the trainer; stop only this recipe's leftovers.
pkill -9 -u "$USER" -f "$code_dir/finetune_${task}.py" || true

ckpt_dir=$(ls -d "$out_dir"/${task}_epoch_*_step_* 2>/dev/null | sort -V | tail -n 1 || true)
if [[ ! -d "$ckpt_dir" ]]; then
    echo "No checkpoint found in $out_dir (see $out_dir/train.log)" >&2
    exit 1
fi
echo "Done. Decode with:"
echo "  ADACTC_CKPT=$ckpt_dir/model.pt bash $code_dir/decode_adactc.sh"
