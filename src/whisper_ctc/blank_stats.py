"""
Blank-token statistics for Whisper-CTC checkpoints (frozen backbone vs partial unfreeze).

Self-contained (does not import train_*.py, which call evaluate.load at import time
and need internet). Model/data definitions mirror train_partial_unfreeze.py.

Usage:
  python blank_stats.py --ckpt models/whisper_ctc_frozen_backbone.pth --tag frozen_backbone \
      --test_jsonl data/librispeech_jsonl/librispeech_test_clean.jsonl \
      --out_dir exp/blank_stats
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import whisper
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from slam_llm.models.encoder import WhisperWrappedEncoder  # noqa: E402

char_to_id = {
    "'": 27, "</s>": 2, "<pad>": 0, "<s>": 1, "<unk>": 3,
    "a": 7, "b": 24, "c": 19, "d": 14, "e": 5, "f": 20, "g": 21, "h": 11,
    "i": 10, "j": 29, "k": 26, "l": 15, "m": 17, "n": 9, "o": 8, "p": 23,
    "q": 30, "r": 13, "s": 12, "t": 6, "u": 16, "v": 25, "w": 18, "x": 28,
    "y": 22, "z": 31, "|": 4,
}
id_to_char = {v: k for k, v in char_to_id.items()}
VOCAB_SIZE, BLANK_ID, SPACE_ID = len(char_to_id), 0, 4
FRAME_SEC = 0.02  # Whisper encoder output frame = 20 ms


class LocalJsonlDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_path, path_remap=None, mel_size=128, raw=False):
        self.data = [json.loads(l) for l in open(jsonl_path, encoding="utf-8")]
        self.path_remap = path_remap or []
        self.mel_size = mel_size
        self.raw = raw

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        src = item["source"]
        for old, new in self.path_remap:
            if src.startswith(old):
                src = new + src[len(old):]
        audio = whisper.load_audio(src)
        labels = [char_to_id[c] for c in item.get("target", "").lower().replace(" ", "|") if c in char_to_id]
        if self.raw:  # wav2vec2: unpadded waveform, per-utterance zero-mean/unit-var (do_normalize=True)
            a = torch.from_numpy(audio)
            a = (a - a.mean()) / torch.sqrt(a.var(unbiased=False) + 1e-7)
            return {"key": item.get("key", str(idx)), "input_features": a, "labels": labels, "mel_len": len(audio)}
        actual_mel_len = min(3000, int(len(audio) / 160))
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio, n_mels=self.mel_size)  # (n_mels, 3000)
        return {"key": item.get("key", str(idx)), "input_features": mel, "labels": labels, "mel_len": actual_mel_len}


def collate(features):
    return {
        "keys": [f["key"] for f in features],
        "input_features": torch.stack([f["input_features"] for f in features]),
        "labels": [f["labels"] for f in features],
        "mel_lens": torch.tensor([f["mel_len"] for f in features]),
    }


class WhisperCTCModel(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.encoder = WhisperWrappedEncoder.load(model_config)
        self.ctc_head = nn.Linear(model_config.encoder_dim, VOCAB_SIZE)

    def forward(self, input_features, mel_lens):
        hidden = self.encoder.extract_variable_length_features(input_features)
        logits = self.ctc_head(hidden)
        b, t_out, _ = logits.shape
        out_lens = (mel_lens + 1) // 2
        mask = torch.arange(t_out, device=logits.device).expand(b, t_out) < out_lens.unsqueeze(1)
        return logits.masked_fill(~mask.unsqueeze(-1), -1e4)


def _edit(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def error_rate(refs, hyps, char=False):
    tok = (lambda s: list(s)) if char else (lambda s: s.split())
    errs = sum(_edit(tok(r), tok(h)) for r, h in zip(refs, hyps))
    return errs / max(1, sum(len(tok(r)) for r in refs))


def runs(mask):
    """Lengths of consecutive True runs in a 1-D bool array."""
    if mask.size == 0:
        return np.array([], dtype=np.int64)
    m = np.concatenate([[False], mask, [False]]).astype(np.int8)
    d = np.diff(m)
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    return ends - starts


def describe(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {}
    p = np.percentile(x, [5, 25, 50, 75, 95])
    return {"mean": x.mean(), "std": x.std(), "min": x.min(), "p5": p[0], "p25": p[1],
            "median": p[2], "p75": p[3], "p95": p[4], "max": x.max(), "n": int(x.size)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--test_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--encoder_path", default="large-v3-turbo")
    ap.add_argument("--remap", nargs=2, action="append", metavar=("OLD", "NEW"),
                    default=[])
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--model_type", choices=["whisper_ctc", "wav2vec2"], default="whisper_ctc",
                    help="wav2vec2: --ckpt is a HF Wav2Vec2ForCTC directory; batch size forced to 1 (no padding).")
    ap.add_argument("--full_window", action="store_true",
                    help="Count over the whole 30 s padded window (1500 frames), no length masking. "
                         "This matches what the SLAM-LLM CTC pruning projector sees.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    is_w2v = args.model_type == "wav2vec2"
    print(f"Loading {args.ckpt}")
    if is_w2v:
        from transformers import Wav2Vec2ForCTC
        model = Wav2Vec2ForCTC.from_pretrained(args.ckpt)
        w2v_vocab = json.load(open(os.path.join(args.ckpt, "vocab.json")))
        assert {k.lower(): v for k, v in w2v_vocab.items()} == char_to_id, "wav2vec2 vocab != whisper-ctc vocab"
        args.batch_size = 1
    else:
        model = WhisperCTCModel(OmegaConf.create({
            "encoder_path": args.encoder_path, "encoder_dim": 1280,
            "encoder_name": "whisper", "whisper_decode": False, "encoder_path_hf": None,
        }))
        model.load_state_dict(torch.load(args.ckpt, map_location="cpu"), strict=True)
    model.to(device).eval()

    ds = LocalJsonlDataset(args.test_jsonl, path_remap=args.remap, raw=is_w2v)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate)

    per_utt = []
    blank_runs_all, nonblank_runs_all = [], []
    pblank_all = []          # p(blank) at every valid frame
    pblank_at_argmax_blank = []
    npz_store = {}

    with torch.no_grad():
        for batch in tqdm(loader):
            x = batch["input_features"].to(device)
            if is_w2v:
                logits = model(x).logits  # (1, T, V), batch size 1 -> all frames valid
                out_lens = np.array([logits.shape[1]])
            else:
                mel_lens = batch["mel_lens"].to(device)
                if args.full_window:
                    mel_lens = torch.full_like(mel_lens, x.shape[-1])
                logits = model(x, mel_lens)  # fp32, same as inference.py
                out_lens = ((mel_lens + 1) // 2).cpu().numpy()
            probs = F.softmax(logits.float(), dim=-1).cpu().numpy()
            pred = probs.argmax(-1)

            for b, key in enumerate(batch["keys"]):
                T = int(out_lens[b])
                ids = pred[b, :T]
                pb = probs[b, :T, BLANK_ID]
                is_blank = ids == BLANK_ID

                # greedy CTC decode
                hyp, prev = [], None
                for v in ids:
                    if v != prev and v != BLANK_ID:
                        hyp.append(id_to_char.get(int(v), ""))
                    prev = v
                hyp = " ".join("".join(hyp).replace("|", " ").split())
                ref = "".join(id_to_char[i] for i in batch["labels"][b]).replace("|", " ").strip()

                br, nbr = runs(is_blank), runs(~is_blank)
                blank_runs_all.append(br)
                nonblank_runs_all.append(nbr)
                pblank_all.append(pb)
                pblank_at_argmax_blank.append(pb[is_blank])

                nz = np.where(~is_blank)[0]
                lead = int(nz[0]) if nz.size else T
                trail = int(T - 1 - nz[-1]) if nz.size else T
                n_nb = int((~is_blank).sum())
                n_ref_chars = len(batch["labels"][b])
                # min frames CTC needs: label length + #repeated adjacent labels
                lbl = batch["labels"][b]
                n_repeats = sum(1 for i in range(1, len(lbl)) if lbl[i] == lbl[i - 1])

                per_utt.append({
                    "key": key, "frames": T, "dur_s": T * FRAME_SEC,
                    "n_blank": int(is_blank.sum()), "n_nonblank": n_nb,
                    "blank_ratio": float(is_blank.mean()) if T else 0.0,
                    "space_frames": int((ids == SPACE_ID).sum()),
                    "n_blank_runs": int(br.size), "max_blank_run": int(br.max()) if br.size else 0,
                    "mean_blank_run": float(br.mean()) if br.size else 0.0,
                    "lead_blanks": lead, "trail_blanks": trail,
                    "inner_blank_ratio": float(is_blank[lead:T - trail].mean()) if T - trail > lead else 0.0,
                    "mean_p_blank": float(pb.mean()) if T else 0.0,
                    "ref_chars": n_ref_chars, "ctc_min_frames": n_ref_chars + n_repeats,
                    "nonblank_per_ref_char": n_nb / max(1, n_ref_chars),
                    "slack_frames": T - (n_ref_chars + n_repeats),
                    "ref": ref, "hyp": hyp,
                })
                npz_store[key] = np.stack([ids.astype(np.float32), pb.astype(np.float32)])

    # ---- aggregate ----
    blank_runs = np.concatenate(blank_runs_all)
    nonblank_runs = np.concatenate(nonblank_runs_all)
    pblank = np.concatenate(pblank_all)
    pblank_ab = np.concatenate(pblank_at_argmax_blank)
    tot_frames = sum(u["frames"] for u in per_utt)
    tot_blank = sum(u["n_blank"] for u in per_utt)
    refs, hyps = [u["ref"] for u in per_utt], [u["hyp"] for u in per_utt]

    summary = {
        "tag": args.tag, "ckpt": args.ckpt, "n_utts": len(per_utt),
        "WER": error_rate(refs, hyps), "CER": error_rate(refs, hyps, char=True),
        "total_frames": tot_frames, "total_blank_frames": tot_blank,
        "global_blank_ratio (frame-weighted)": tot_blank / tot_frames,
        "utt_blank_ratio (macro avg, = inference.py 'Average Blank Density')": describe([u["blank_ratio"] for u in per_utt]),
        "utt_inner_blank_ratio (excl. leading/trailing silence)": describe([u["inner_blank_ratio"] for u in per_utt]),
        "lead_blank_frames": describe([u["lead_blanks"] for u in per_utt]),
        "trail_blank_frames": describe([u["trail_blanks"] for u in per_utt]),
        "blank_run_length (frames)": describe(blank_runs),
        "nonblank_run_length (frames)": describe(nonblank_runs),
        "frac_nonblank_runs_len1": float((nonblank_runs == 1).mean()),
        "nonblank_frames_per_ref_char": describe([u["nonblank_per_ref_char"] for u in per_utt]),
        "space('|')_frame_ratio": sum(u["space_frames"] for u in per_utt) / tot_frames,
        "p_blank_all_frames": describe(pblank),
        "p_blank_on_argmax_blank_frames": describe(pblank_ab),
        "frac_frames_p_blank_gt": {str(t): float((pblank > t).mean()) for t in [0.5, 0.9, 0.95, 0.99, 0.999]},
        "blank_ratio_hist (utt, 10 bins 0-1)": np.histogram([u["blank_ratio"] for u in per_utt],
                                                            bins=10, range=(0, 1))[0].tolist(),
        "blank_run_len_hist (1,2,3,4,5,6-10,11-25,26-50,51+)": np.histogram(
            blank_runs, bins=[1, 2, 3, 4, 5, 6, 11, 26, 51, 10**6])[0].tolist(),
        "corr(blank_ratio, dur_s)": float(np.corrcoef([u["blank_ratio"] for u in per_utt],
                                                       [u["dur_s"] for u in per_utt])[0, 1]),
    }

    def conv(o):
        if isinstance(o, dict):
            return {k: conv(v) for k, v in o.items()}
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o

    summary = conv(summary)
    with open(os.path.join(args.out_dir, f"{args.tag}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out_dir, f"{args.tag}_per_utt.jsonl"), "w") as f:
        for u in per_utt:
            f.write(json.dumps(u) + "\n")
    np.savez_compressed(os.path.join(args.out_dir, f"{args.tag}_frames.npz"), **npz_store)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
