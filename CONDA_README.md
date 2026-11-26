# Setting up a conda environment for AdaCTC

These steps reproduce the reference environment: Python 3.12.3, PyTorch 2.9.1 with CUDA 12.8 wheels, and transformers 4.57.6. They were tested with miniforge3 on an HPC cluster whose GPU nodes have no internet access. Run every download step on a login node.

## 1. Check CUDA

```bash
module load cuda    # on clusters with environment modules
nvidia-smi          # the driver must support CUDA 12.x
```

## 2. Create the environment

```bash
conda create -n adactc python=3.12.3 -y
conda activate adactc

pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir \
    torch==2.9.1 \
    torchvision==0.24.1 \
    torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu128

cd /path/to/AdaCTC
pip install --no-cache-dir -r requirements.txt
conda install -c conda-forge ffmpeg -y    # whisper.load_audio decodes audio with ffmpeg
```

## 3. Download the models

All scripts expect the models in `models/` at the repository root.

```bash
cd /path/to/AdaCTC
mkdir -p models

# LLM backbone
hf download utter-project/EuroLLM-1.7B --local-dir models/eurollm-1.7b
# The repository ships pytorch_model.bin; store it as safetensors so transformers loads it without torch.load
python -c "import torch; from safetensors.torch import save_file; p='models/eurollm-1.7b/pytorch_model.bin'; save_file(torch.load(p, map_location='cpu', weights_only=False), p.replace('pytorch_model.bin', 'model.safetensors'))"
mv models/eurollm-1.7b/pytorch_model.bin models/eurollm-1.7b/pytorch_model.bin.bak

# Whisper large-v3-turbo (the Whisper-CTC encoder is built from it)
wget -P models https://openaipublic.azureedge.net/main/whisper/models/aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a/large-v3-turbo.pt
mkdir -p ~/.cache/whisper && cp models/large-v3-turbo.pt ~/.cache/whisper/

# Whisper CTC heads and Adaptive checkpoints
# Replace <hf-org> with the Hugging Face organisation that hosts the checkpoints
hf download <hf-org>/AdaCTC --local-dir models
#   -> models/whisper_ctc_frozen_backbone.pth
#      models/whisper_ctc_partial_unfreeze.pth
#      models/adactc_asr_voxpopuli/model.pt
#      models/adactc_st_europarl/model.pt
```

## 4. Cache the evaluation metrics

`evaluate` downloads its metric scripts the first time they are loaded. Load them once on a login node so offline GPU jobs find them in the cache:

```bash
python -c "import evaluate; evaluate.load('wer'); evaluate.load('bleu')"
```

Then export `HF_HUB_OFFLINE=1` in your GPU jobs.

## 5. Data

Put the jsonl manifests under `data/`, or set `DATA_DIR` (see the README for the expected layout). Every manifest stores absolute audio paths in its `source` field. If you move the audio, rewrite those paths, for example:

```bash
sed -i 's|/old/corpora/root|/new/corpora/root|g' data/*/*.jsonl
```

## 6. Check the installation

```bash
cd /path/to/AdaCTC
PYTHONPATH=src python -c "import whisper, slam_llm.models.slam_model, slam_llm.pipeline.inference_batch; print('ok')"
```
