# AdaCTC

Code and models for **Adaptive CTC-Guided Token Pruning for Budget-Aware SpeechLLMs** (EACL 2027).

AdaCTC shortens the speech-token sequence that a Whisper encoder passes to an LLM. An auxiliary CTC head scores every encoder frame, and frames that the head confidently labels as CTC blank are dropped before the projector. The **Adaptive** model is trained with a pruning threshold sampled per batch, so a single checkpoint can be decoded at any token rate (tokens per second, TPS) between 10 and the CTC rate.

This repository contains:

- **Whisper CTC heads**: training, evaluation and blank-statistics code, plus the two checkpoints (frozen backbone and partially fine-tuned).
- **Adaptive SLLM checkpoints** for ASR (VoxPopuli-EN) and ST (Europarl-ST de/fr→en), with their training code and the inference code for CTC-guided pruning.
- **Fixed-threshold CTC pruning (ds=5)**: training and inference code for ASR and ST, one model per threshold.

It is a trimmed-down derivative of [SLAM-LLM](https://github.com/X-LANCE/SLAM-LLM).

## Layout

```
src/whisper_ctc/          CTC head: training, test-set WER, blank statistics
src/slam_llm/             SLAM-LLM core (Whisper-CTC encoder, pruning projector, training/inference pipelines)
src/slam_llm/utils/plot_{wer,cm,flops}.py   paper figures
examples/asr_voxpopuli/   ASR recipe: train_adactc.sh, decode_adactc.sh, ctc_prune_ds5.sh
examples/st_europarl/     ST recipe: train_adactc.sh, decode_adactc.sh, ctc_prune_ds5.sh
scripts/data_prep/        Europarl-ST segmentation into jsonl manifests
```

## Setup

```bash
conda create -n adactc python=3.12.3 -y && conda activate adactc
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
conda install -c conda-forge ffmpeg -y
```

See [CONDA_README.md](CONDA_README.md) for the full setup: model downloads, and caching the evaluation metrics for offline GPU nodes.

A `Dockerfile` is also provided. Scripts take the repository root from `git rev-parse` (or the current directory) and read data from `$DATA_DIR` (default `<repo>/data`). Both can be overridden with `ADACTC_ROOT` and `DATA_DIR`.

### Models

Download the checkpoints into `models/`:

| Path | Contents | Source |
|---|---|---|
| `whisper_ctc_frozen_backbone.pth` | Whisper large-v3-turbo encoder + CTC head (frozen backbone) | [Hugging Face](https://huggingface.co/<hf-org>/AdaCTC) |
| `whisper_ctc_partial_unfreeze.pth` | Whisper large-v3-turbo encoder + CTC head (last 4 blocks fine-tuned) | [Hugging Face](https://huggingface.co/<hf-org>/AdaCTC) |
| `adactc_asr_voxpopuli/model.pt` | Adaptive ASR model: projector + LoRA weights | [Hugging Face](https://huggingface.co/<hf-org>/AdaCTC) |
| `adactc_st_europarl/model.pt` | Adaptive ST model: projector + LoRA weights | [Hugging Face](https://huggingface.co/<hf-org>/AdaCTC) |
| `eurollm-1.7b/` | LLM backbone | [utter-project/EuroLLM-1.7B](https://huggingface.co/utter-project/EuroLLM-1.7B) |

The Adaptive checkpoints hold only the trained parameters. The Whisper encoder comes from `whisper_ctc_partial_unfreeze.pth`, and EuroLLM-1.7B is loaded separately.

### Data

Every script reads jsonl manifests with one utterance per line:

```json
{"key": "utt-id", "source": "/abs/path/to/audio.wav", "target": "reference transcript or translation"}
```

| Used by | Manifest |
|---|---|
| CTC head training/eval | `$DATA_DIR/librispeech_jsonl/{train-libri,librispeech_dev_other,librispeech_test_clean}.jsonl` |
| ASR | `$DATA_DIR/voxpopuli_jsonl/{train,val,test}.jsonl` (decoding only needs `test`) |
| ST | `$DATA_DIR/europarl_de2en_fr2en_jsonl/{train,val,test}.jsonl` (see `scripts/data_prep/preprocess_europarl.py`) |

The datasets are not redistributed here.

## Whisper CTC heads

Run these from the repository root:

```bash
export PYTHONPATH=src
python src/whisper_ctc/train_frozen_backbone.py     # linear head only (lr 1e-4, cosine)
python src/whisper_ctc/train_partial_unfreeze.py    # last 4 encoder blocks + head, SpecAugment, entropy regulariser
python src/whisper_ctc/inference.py                 # greedy CTC decoding, WER on LibriSpeech test-clean
python src/whisper_ctc/blank_stats.py --ckpt models/whisper_ctc_partial_unfreeze.pth --tag partial_unfreeze \
    --test_jsonl "${DATA_DIR:-data}/librispeech_jsonl/librispeech_test_clean.jsonl" --out_dir exp/blank_stats
```

The training scripts write to `exp/whisper_ctc_{frozen_backbone,partial_unfreeze}/`. `inference.py` evaluates `models/whisper_ctc_frozen_backbone.pth`; edit the path in its `__main__` block to evaluate the other head.

Both heads are trained on LibriSpeech 960 h, and the checkpoint with the best dev-other WER is kept. The `slurm_blank_stats*.sh` scripts are the SLURM jobs used for the blank-statistics analysis. Set `#SBATCH --account`, `CONDA_PATH` and `CONDA_ENV` for your cluster, and create `exp/blank_stats/` before submitting.

## Adaptive (AdaCTC) training

```bash
bash examples/asr_voxpopuli/train_adactc.sh    # 10 epochs on VoxPopuli-EN
bash examples/st_europarl/train_adactc.sh      # 30 epochs on Europarl-ST
```

A single model is trained for all token rates. For every utterance, the CTC-blank threshold is drawn at random from the set at the top of the script, which is the same set `decode_adactc.sh` evaluates. The Whisper-CTC encoder and EuroLLM-1.7B stay frozen. Only the projector and LoRA adapters are trained, on one GPU with fp16, batch size 4, lr 1e-4 and 1000 warm-up steps. The script prints the checkpoint path at the end. Pass it to the decoder with `ADACTC_CKPT=<path>/model.pt`.

## Adaptive inference with CTC-guided pruning

```bash
bash examples/asr_voxpopuli/decode_adactc.sh    # WER per token rate
bash examples/st_europarl/decode_adactc.sh      # BLEU per token rate
```

Each script decodes the test set once per CTC-blank threshold and writes `exp/adactc_*/thr<threshold>/{decode_test_pred,decode_test_gt,WER.txt|BLEU.txt}`. A frame is dropped when its most likely label is blank and its blank probability is at least the threshold. So `1.1` keeps every frame (10 TPS) and `0.0` removes every blank-labelled frame. The thresholds that give the token rates reported in the paper are listed at the top of each script. Pass a subset with, for example, `THRESHOLDS="1.1 0.0"`. Paths can be overridden with `LLM_PATH`, `ENCODER_PATH`, `ADACTC_CKPT`, `TEST_JSONL` and `OUT_DIR`.

## CTC pruning at ds=5 (one model per threshold)

```bash
bash examples/asr_voxpopuli/ctc_prune_ds5.sh    # train + decode + WER, per threshold
bash examples/st_europarl/ctc_prune_ds5.sh      # train + decode + BLEU, per threshold
```

For each threshold, the script trains the projector and LoRA adapters with that fixed threshold. The Whisper-CTC encoder and EuroLLM-1.7B stay frozen. It then decodes the test set with the same threshold and writes everything to `exp/ctc_prune_ds5_*/thr<threshold>/`. Training uses one GPU, fp16, batch size 4, lr 1e-4, 1000 warm-up steps, and 4 epochs (ASR) or 10 epochs (ST). The latest checkpoint is used for decoding. The thresholds are the same as in `decode_adactc.sh`, and `THRESHOLDS`, `JSONL_DIR`, `LLM_PATH`, `ENCODER_PATH` and `OUT_DIR` can be overridden.

Weights & Biases logging is off. Add `++log_config.use_wandb=true` to the training command to turn it on. Entity and project then come from `WANDB_ENTITY` / `WANDB_PROJECT`, or from `++log_config.wandb_entity_name=...` / `++log_config.wandb_project_name=...`.

## Figures

```bash
python src/slam_llm/utils/plot_wer.py     # metric vs. token rate, all tasks and methods
python src/slam_llm/utils/plot_cm.py      # train/test token-rate mismatch heatmaps (WER, BLEU)
python src/slam_llm/utils/plot_flops.py   # GFLOPs vs. token rate (theoretical, see below)
```

The paper's results are hardcoded in the scripts, and the figures are written to the current directory as PDF and PNG.

## Citation

```bibtex
@inproceedings{kadandale2027adaptive,
  title     = {Adaptive {CTC}-Guided Token Pruning for Budget-Aware {SpeechLLMs}},
  author    = {Kadandale, Venkatesh S. and Nabih, Mohamed and Falavigna, Daniele and Brutti, Alessio},
  booktitle = {Proceedings of the 2027 Conference of the European Chapter of the Association for Computational Linguistics (EACL)},
  year      = {2027},
  note      = {To appear}
}
```

## Acknowledgements

This work was supported by Ministero delle Imprese e del Made in Italy (IPCEI Cloud DM 27 giugno 2022 – IPCEI-CL-0000007) and European Union (Next Generation EU). We acknowledge CINECA for the availability of high-performance computing resources and support. The authors would like to thank Leonardo Badino and Giovanni Morrone of Almawave for their insightful discussions and valuable collaboration.

## Licence

Built on [SLAM-LLM](https://github.com/X-LANCE/SLAM-LLM) (MIT, © 2024 Ziyang Ma) and [OpenAI Whisper](https://github.com/openai/whisper) (MIT). The code is released under the MIT licence (see `LICENSE`). Model checkpoints carry the licence stated on their model cards. The ST checkpoint is trained on Europarl-ST (CC BY-NC 4.0).
