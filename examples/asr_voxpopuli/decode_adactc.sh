#!/bin/bash
# Decode VoxPopuli-EN test with the Adaptive (AdaCTC) ASR checkpoint at several token rates and score WER.
#
# One checkpoint serves every rate: the CTC-blank threshold is chosen at inference time.
# A frame is dropped when its argmax is the CTC blank AND its blank probability is >= the threshold,
# so 1.1 keeps every frame and 0.0 keeps only non-blank frames.
#
# Overridable via environment: ADACTC_ROOT, DATA_DIR, LLM_PATH, ENCODER_PATH, ADACTC_CKPT,
# TEST_JSONL, OUT_DIR, THRESHOLDS (space-separated subset of the list below), CUDA_VISIBLE_DEVICES.
set -euo pipefail

export ADACTC_ROOT="${ADACTC_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
export DATA_DIR="${DATA_DIR:-$ADACTC_ROOT/data}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTHONPATH="$ADACTC_ROOT/src"
cd "$ADACTC_ROOT"

code_dir=examples/asr_voxpopuli
llm_path="${LLM_PATH:-$ADACTC_ROOT/models/eurollm-1.7b}"
encoder_path="${ENCODER_PATH:-$ADACTC_ROOT/models/whisper_ctc_partial_unfreeze.pth}"
ckpt="${ADACTC_CKPT:-$ADACTC_ROOT/models/adactc_asr_voxpopuli/model.pt}"
test_jsonl="${TEST_JSONL:-$DATA_DIR/voxpopuli_jsonl/test.jsonl}"
out_root="${OUT_DIR:-$ADACTC_ROOT/exp/adactc_asr_voxpopuli}"

# Threshold -> token rate (tokens/s) on VoxPopuli-EN test, with the partially fine-tuned CTC head:
#   1.1 -> 10 | 0.9999951124191284 -> 8.34 | 0.999987006187439 -> 7.14 | 0.9999709129333496 -> 6.25
#   0.9999358952045441 -> 5.56 | 0.9998627007007599 -> 5 | 0.9931716918945312 -> 3.34
#   0.89990234375 -> 2.5 | 0.0 -> 1.72 (all blank-argmax frames removed)
default_thresholds="1.1 0.9999951124191284 0.999987006187439 0.9999709129333496 0.9999358952045441 0.9998627007007599 0.9931716918945312 0.89990234375 0.0"
read -r -a thresholds <<< "${THRESHOLDS:-$default_thresholds}"

for f in "$llm_path" "$encoder_path" "$ckpt" "$test_jsonl"; do
    [[ -e "$f" ]] || { echo "Missing: $f" >&2; exit 1; }
done

for thr in "${thresholds[@]}"; do
    out_dir="$out_root/thr${thr}"
    mkdir -p "$out_dir"
    decode_log="$out_dir/decode_test"
    echo ">> threshold $thr -> $out_dir"

    python -u "$code_dir/inference_asr_batch.py" \
        --config-path "conf" --config-name "prompt.yaml" \
        hydra.run.dir="$out_dir" \
        ++model_config.llm_name=eurollm-1.7b \
        ++model_config.llm_path="$llm_path" \
        ++model_config.llm_dim=2048 \
        ++model_config.encoder_name=whisper-ctc \
        ++model_config.encoder_path="$encoder_path" \
        ++model_config.encoder_dim=1280 \
        ++model_config.encoder_ctc_layer=32 \
        ++model_config.ctc_vocab_size=-1 \
        ++model_config.encoder_projector=pruning \
        ++model_config.encoder_projector_ds_rate=5 \
        ++model_config.downsampling_type=linear \
        ++model_config.pruning_type=ctc \
        ++model_config.encoder_prune_thr="$thr" \
        ++model_config.normalize=true \
        ++dataset_config.dataset=speech_dataset \
        ++dataset_config.input_type=mel \
        ++dataset_config.mel_size=128 \
        ++dataset_config.val_data_path="$test_jsonl" \
        ++dataset_config.inference_mode=true \
        ++train_config.model_name=asr \
        ++train_config.freeze_encoder=true \
        ++train_config.freeze_llm=true \
        ++train_config.use_peft=true \
        ++train_config.batching_strategy=custom \
        ++train_config.num_epochs=1 \
        ++train_config.val_batch_size=1 \
        ++decode_log="$decode_log" \
        ++ckpt_path="$ckpt" \
        > "$out_dir/inference.log" 2>&1

    python src/slam_llm/utils/compute_wer.py "${decode_log}_gt" "${decode_log}_pred" "$out_dir/WER.txt"
done
