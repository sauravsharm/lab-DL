import json
import math
import os
import re
from collections import Counter
from typing import Any, List, Optional, Tuple, Dict
from datetime import datetime


def simple_tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"([.,!?;:()\"'])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


def load_poems(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_vocab(tokens: List[str], min_freq: int = 1) -> Tuple[Dict[str, int], Dict[int, str]]:
    counts = Counter(tokens)
    vocab = ["<pad>", "<unk>", "<bos>", "<eos>"]
    for w, c in counts.items():
        if c >= min_freq and w not in vocab:
            vocab.append(w)
    stoi = {w: i for i, w in enumerate(vocab)}
    itos = {i: w for w, i in stoi.items()}
    return stoi, itos


def tokens_to_ids(tokens: List[str], stoi: Dict[str, int]) -> List[int]:
    unk = stoi["<unk>"]
    return [stoi.get(t, unk) for t in tokens]


def make_sequences(ids: List[int], seq_len: int) -> Tuple[List[List[int]], List[List[int]]]:
    X, Y = [], []
    for i in range(0, len(ids) - seq_len):
        x = ids[i:i+seq_len]
        y = ids[i+1:i+seq_len+1]
        X.append(x)
        Y.append(y)
    return X, Y


def text_quality_metrics(text: str) -> Dict[str, Any]:
    toks = simple_tokenize(text)
    if not toks:
        return {
            "token_count": 0,
            "unique_tokens": 0,
            "unique_ratio": 0.0,
            "repeat_2gram_ratio": 0.0,
            "repeat_3gram_ratio": 0.0,
            "top_tokens": [],
        }

    counts = Counter(toks)
    token_count = len(toks)
    unique_tokens = len(counts)
    unique_ratio = unique_tokens / max(token_count, 1)

    def repeat_ngram_ratio(n: int) -> float:
        if len(toks) < n:
            return 0.0
        ngrams = list(zip(*[toks[i:] for i in range(n)]))
        c = Counter(ngrams)
        repeats = sum(v - 1 for v in c.values() if v > 1)
        return repeats / max(len(ngrams), 1)

    return {
        "token_count": int(token_count),
        "unique_tokens": int(unique_tokens),
        "unique_ratio": float(unique_ratio),
        "repeat_2gram_ratio": float(repeat_ngram_ratio(2)),
        "repeat_3gram_ratio": float(repeat_ngram_ratio(3)),
        "top_tokens": [(w, int(c)) for w, c in counts.most_common(10)],
    }


def save_run_log(
    out_path: str,
    run_name: str,
    epoch_losses: List[float],
    epoch_times: List[float],
    samples: List[str],
    config: Optional[Dict[str, Any]] = None,
    notes: Optional[str] = None,
    sample_for_quality: str = "last",
) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    chosen_sample = ""
    if samples:
        if sample_for_quality == "last":
            chosen_sample = samples[-1]
        elif sample_for_quality == "best_loss" and epoch_losses:
            best_idx = min(range(len(epoch_losses)),
                           key=lambda i: epoch_losses[i])
            chosen_sample = samples[min(best_idx, len(samples) - 1)]
        elif sample_for_quality.startswith("index:"):
            try:
                k = int(sample_for_quality.split(":", 1)[1])
                chosen_sample = samples[max(0, min(k, len(samples) - 1))]
            except Exception:
                chosen_sample = samples[-1]
        else:
            chosen_sample = samples[-1]

    payload = {
        "run_name": run_name,
        "epoch_losses": [float(x) for x in epoch_losses],
        "epoch_times": [float(x) for x in epoch_times],
        "samples": list(samples),
        "quality_on_sample": sample_for_quality,
        "quality_metrics": text_quality_metrics(chosen_sample) if chosen_sample else text_quality_metrics(""),
        "config": config or {},
        "notes": notes or "",
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return out_path


def load_run(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize(run: Dict[str, Any]) -> Dict[str, Any]:
    times = run.get("epoch_times", [])
    losses = run.get("epoch_losses", [])
    qm = run.get("quality_metrics", {}) or {}

    total_time = float(sum(times)) if times else float("nan")
    avg_epoch_time = float(total_time / len(times)) if times else float("nan")

    final_loss = float(losses[-1]) if losses else float("nan")
    best_loss = float(min(losses)) if losses else float("nan")

    return {
        "run_name": run.get("run_name", "run"),
        "total_time": total_time,
        "avg_epoch_time": avg_epoch_time,
        "final_loss": final_loss,
        "best_loss": best_loss,
        "token_count": qm.get("token_count", 0),
        "unique_ratio": qm.get("unique_ratio", float("nan")),
        "repeat_2gram_ratio": qm.get("repeat_2gram_ratio", float("nan")),
        "repeat_3gram_ratio": qm.get("repeat_3gram_ratio", float("nan")),
        "top_tokens": qm.get("top_tokens", []),
    }


def fmt(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    return f"{x:.4f}"


def compare_runs(runs_path):
    runs = [load_run(r) for r in runs_path]
    sums = [summarize(r) for r in runs]
    res = ""

    res = "Comparison and Analysis"
    res += "\nTraining time and loss:"
    for s, p in zip(sums, runs_path):
        res += f"{s['run_name']:<18} | total_time={fmt(s['total_time'])}s | avg_epoch={fmt(s['avg_epoch_time'])}s | best_loss={fmt(s['best_loss'])} | final_loss={fmt(s['final_loss'])} | file={p}"

    res += "\nGenerated text quality (heuristics on saved sample):"
    for s in sums:
        res += f"{s['run_name']:<18} | tokens={s['token_count']:<4} | unique_ratio={fmt(float(s['unique_ratio']))} | repeat_2gram={fmt(float(s['repeat_2gram_ratio']))} | repeat_3gram={fmt(float(s['repeat_3gram_ratio']))}"

    res += "\nTop tokens (for quick sanity check):"
    for s in sums:
        res += f"{s['run_name']:<18} | {s['top_tokens'][:8]}"

    print(res)
    data = {}
    if os.path.exists("results.json"):
        with open("results.json", "r") as f:
            data = json.load(f)

    data[f"results_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"] = res
    with open("results.json", "w") as f:
        json.dump(data, f, indent=2)
