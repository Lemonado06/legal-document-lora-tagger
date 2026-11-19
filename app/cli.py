"""Command line entry point.

    python -m app.cli prepare  --source synthetic            # or ledgar / path to csv|jsonl
    python -m app.cli train    --data data/clauses.jsonl --out runs/r8 --rank 8
    python -m app.cli evaluate --run runs/r8
    python -m app.cli merge    --run runs/r8
    python -m app.cli tag      --run runs/r8 --input examples/sample_contract.txt
    python -m app.cli sweep    --data data/clauses.jsonl --ranks 2 4 8 16

Add --tiny to train/sweep to use a small random GPT-2 (no download, runs in seconds).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

from . import core, dataset, llm, storage


def _load_env(path: str = ".env") -> None:
    """Tiny .env reader so we don't need python-dotenv."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# -------------------------------------------------------------------- hello --

def cmd_hello(a):
    """Sanity check: shows the device, model and that torch/transformers import."""
    import transformers
    print("legal-document-lora-tagger is installed")
    print(f"  torch {torch.__version__}, transformers {transformers.__version__}")
    print(f"  device: {llm.get_device()}  model: {os.getenv('MODEL_NAME', 'gpt2')}")
    print(f"  synthetic tags: {', '.join(dataset.LABELS)}")
    print("next: python -m app.cli prepare --source synthetic")


# ------------------------------------------------------------------ prepare --

def cmd_prepare(a) -> Path:
    if a.source == "synthetic":
        rows = dataset.make_synthetic(n_per_label=a.per_label, seed=a.seed)
    elif a.source == "ledgar":
        rows = dataset.load_ledgar(top_k=a.top_k, per_label=a.per_label, seed=a.seed)
    else:
        rows = dataset.read_rows(a.source)
    out = dataset.write_jsonl(rows, a.out)
    counts = {}
    for r in rows:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    print(f"wrote {len(rows)} clauses, {len(counts)} tags -> {out}")
    for k, v in sorted(counts.items()):
        print(f"  {k:<28} {v}")
    return out


# -------------------------------------------------------------------- train --

def train_run(data: str, out: str, rank: int = 8, alpha: float | None = None, epochs: int = 3,
              batch_size: int = 8, lr: float = 2e-4, emb_lr: float = 5e-3, targets=("c_attn",), tiny: bool = False,
              full: bool = False, model_name: str | None = None, max_len: int = 256,
              device: str | None = None, seed: int = 13, log=print) -> dict:
    rows = dataset.read_rows(data)
    train_rows, val_rows, test_rows = dataset.stratified_split(rows, seed=seed)
    labels = sorted({r["label"] for r in rows})
    model_name = model_name or os.getenv("MODEL_NAME", "gpt2")
    alpha = alpha if alpha is not None else 2 * rank
    dev = llm.get_device(device)

    tok, model, info = llm.build_model(labels, model_name, tiny=tiny,
                                       corpus=[r["text"] for r in train_rows])
    label2id = dict(zip(labels, info["label_token_ids"]))
    run_dir = Path(out)
    run_dir.mkdir(parents=True, exist_ok=True)
    if tiny:
        storage.save_tiny_base(run_dir, model)

    if full:
        for p in model.parameters():
            p.requires_grad = True
        replaced = []
    else:
        replaced = core.inject_lora(model, targets=targets, r=rank, alpha=alpha, dropout=0.05)
        core.train_new_token_rows(model, info["old_vocab_size"])
    params = core.count_params(model, 0 if full else info["old_vocab_size"])
    log(f"device={dev}  base={'tiny-gpt2' if tiny else model_name}  "
        f"{'full fine-tune' if full else f'LoRA r={rank} alpha={alpha} on {len(replaced)} layers'}")
    log(f"trainable params: {params['trainable']:,} / {params['total']:,} ({params['trainable_pct']:.3f}%)")
    log(f"split: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")

    cfg = core.TrainConfig(epochs=epochs, batch_size=batch_size, emb_lr=emb_lr, lr=lr if (tiny or not full) else min(lr, 5e-5),
                           max_len=max_len, seed=seed, full_finetune=full)
    result = core.train(model, tok, train_rows, val_rows, label2id, cfg, dev, log=log)

    run_cfg = {"labels": labels, "base_model": model_name, "tiny": tiny, "full_finetune": full,
               "rank": rank, "alpha": alpha, "targets": list(targets), "old_vocab_size": info["old_vocab_size"],
               "max_len": max_len, "epochs": epochs, "batch_size": batch_size, "lr": cfg.lr, "emb_lr": emb_lr, "seed": seed,
               "data": str(data)}
    saved = storage.save_run(run_dir, model, tok, run_cfg)
    dataset.write_jsonl(test_rows, run_dir / "test.jsonl")
    dataset.write_jsonl(val_rows, run_dir / "val.jsonl")
    summary = {"params": params, "train_seconds": result.seconds, "steps": result.steps,
               "sec_per_step": result.sec_per_step, "peak_memory_mb": result.peak_memory_mb,
               "history": result.history, **saved, "device": str(dev)}
    storage.save_json(summary, run_dir / "train_summary.json")
    log(f"saved run -> {run_dir}  (adapter {saved['adapter_bytes'] / 2**20:.2f} MB, "
        f"{result.seconds:.1f}s, {result.sec_per_step * 1000:.0f} ms/step)")
    return summary


def cmd_train(a):
    return train_run(a.data, a.out, rank=a.rank, alpha=a.alpha, epochs=a.epochs, batch_size=a.batch_size,
                     lr=a.lr, emb_lr=a.emb_lr, targets=tuple(a.targets), tiny=a.tiny, full=a.full, model_name=a.model,
                     max_len=a.max_len, device=a.device, seed=a.seed)


# ----------------------------------------------------------------- evaluate --

def evaluate_run(run: str, device: str | None = None, latency_n: int = 50, log=print) -> dict:
    run_dir = Path(run)
    dev = llm.get_device(device)
    tok, model, cfg, label2id = storage.load_run(run_dir)
    model.to(dev)
    test = dataset.read_rows(run_dir / "test.jsonl")
    texts, gold = [r["text"] for r in test], [r["label"] for r in test]

    preds = core.predict(model, tok, texts, label2id, dev, max_len=cfg["max_len"])
    report = core.classification_report(gold, [p["label"] for p in preds], cfg["labels"])
    sample = texts[:latency_n]
    ms_unmerged = core.time_inference(model, tok, sample, label2id, dev)

    merged_layers = 0 if cfg.get("full_finetune") else core.merge_lora(model)
    preds_m = core.predict(model, tok, texts, label2id, dev, max_len=cfg["max_len"])
    ms_merged = core.time_inference(model, tok, sample, label2id, dev)
    agree = sum(p["label"] == q["label"] for p, q in zip(preds, preds_m)) / max(1, len(preds))
    max_diff = max((abs(p["scores"][l] - q["scores"][l]) for p, q in zip(preds, preds_m) for l in p["scores"]),
                   default=0.0)

    ts = storage.load_json(run_dir / "train_summary.json")
    total = ts["params"]["total"]
    trainable = ts["params"]["trainable"]
    full_bytes = total * 4
    efficiency = {
        "trainable_params": trainable, "total_params": total, "trainable_pct": ts["params"]["trainable_pct"],
        "adapter_mb": ts["adapter_bytes"] / 2**20, "full_checkpoint_mb_fp32": full_bytes / 2**20,
        "storage_saving_x": full_bytes / max(1, ts["adapter_bytes"]),
        # AdamW keeps 2 fp32 moments + 1 fp32 grad per trainable param
        "optimizer_state_mb_lora": trainable * 12 / 2**20, "optimizer_state_mb_full_ft": total * 12 / 2**20,
        "train_seconds": ts["train_seconds"], "sec_per_step": ts["sec_per_step"],
        "peak_memory_mb": ts["peak_memory_mb"],
        "latency_ms_per_clause_unmerged": ms_unmerged, "latency_ms_per_clause_merged": ms_merged,
        "merge_prediction_agreement": agree, "merge_max_prob_diff": max_diff, "merged_layers": merged_layers,
    }
    mistakes = [{"text": t, "gold": g, "pred": p["label"], "confidence": round(p["confidence"], 3)}
                for t, g, p in zip(texts, gold, preds) if g != p["label"]]
    metrics = {"tagging": report, "efficiency": efficiency, "config": cfg, "mistakes": mistakes[:50]}
    storage.save_json(metrics, run_dir / "metrics.json")
    (run_dir / "report.md").write_text(render_report(metrics), encoding="utf-8")
    log(f"accuracy {report['accuracy']:.4f}  macro-F1 {report['macro_f1']:.4f}  (n={report['n']})")
    log(f"latency {ms_unmerged:.2f} ms unmerged -> {ms_merged:.2f} ms merged; "
        f"merge agreement {agree:.3f}, max prob diff {max_diff:.2e}")
    log(f"report -> {run_dir / 'report.md'}")
    return metrics


def render_report(m: dict) -> str:
    t, e, c = m["tagging"], m["efficiency"], m["config"]
    kind = "Full fine-tune" if c.get("full_finetune") else f"LoRA r={c['rank']}, alpha={c['alpha']}, targets={c['targets']}"
    lines = [f"# Evaluation: {kind}", "",
             f"Base: `{'tiny-gpt2 (random init)' if c.get('tiny') else c['base_model']}`, data: `{c['data']}`", "",
             "## Tagging", "",
             f"- Accuracy: **{t['accuracy']:.4f}** on {t['n']} held-out clauses",
             f"- Macro F1: **{t['macro_f1']:.4f}**, weighted F1: {t['weighted_f1']:.4f}", "",
             "| Tag | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
    for lab, v in sorted(t["per_label"].items(), key=lambda x: x[1]["f1"]):
        lines.append(f"| `{lab}` | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {v['support']} |")
    labs = list(t["confusion"])
    short = [l[:10] for l in labs]
    lines += ["", "### Confusion matrix (rows = true, cols = predicted)", "",
              "| | " + " | ".join(short) + " |", "|" + "---|" * (len(labs) + 1)]
    for l in labs:
        lines.append(f"| {l[:10]} | " + " | ".join(str(t["confusion"][l][p]) for p in labs) + " |")
    pm = e["peak_memory_mb"]
    lines += ["", "## Efficiency", "",
              f"- Trainable params: {e['trainable_params']:,} of {e['total_params']:,} (**{e['trainable_pct']:.3f}%**)",
              f"- Adapter on disk: {e['adapter_mb']:.2f} MB vs {e['full_checkpoint_mb_fp32']:.1f} MB for a full fp32 checkpoint "
              f"(~{e['storage_saving_x']:.0f}x smaller)",
              f"- Grad + AdamW state: ~{e['optimizer_state_mb_lora']:.1f} MB with LoRA vs ~{e['optimizer_state_mb_full_ft']:.1f} MB for full fine-tuning",
              f"- Training: {e['train_seconds']:.1f}s total, {e['sec_per_step'] * 1000:.0f} ms/step"
              + (f", peak memory {pm:.0f} MB" if pm else ""),
              f"- Inference (batch 1): {e['latency_ms_per_clause_unmerged']:.2f} ms/clause with adapter, "
              f"{e['latency_ms_per_clause_merged']:.2f} ms/clause after merge",
              f"- Merge check: {e['merge_prediction_agreement'] * 100:.1f}% identical predictions, "
              f"max probability drift {e['merge_max_prob_diff']:.2e}", ""]
    if m["mistakes"]:
        lines += ["## Some mistakes", ""]
        for x in m["mistakes"][:8]:
            lines.append(f"- gold `{x['gold']}`, predicted `{x['pred']}` ({x['confidence']}): {x['text'][:160]}")
    return "\n".join(lines) + "\n"


def cmd_evaluate(a):
    return evaluate_run(a.run, device=a.device)


# -------------------------------------------------------------- merge / tag --

def cmd_merge(a):
    tok, model, cfg, _ = storage.load_run(a.run)
    n = core.merge_lora(model)
    out = Path(a.run) / "merged"
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"merged {n} LoRA layers -> {out} ({storage.dir_size(out) / 2**20:.1f} MB)")


def cmd_tag(a):
    dev = llm.get_device(a.device)
    tok, model, cfg, label2id = storage.load_run(a.run, merged=True)
    if not cfg.get("full_finetune") and core.lora_layers(model):
        core.merge_lora(model)
    model.to(dev)
    text = Path(a.input).read_text(encoding="utf-8")
    clauses = dataset.split_document(text)
    tagged = core.tag_document(model, tok, clauses, label2id, dev)
    out = a.out or str(Path(a.run) / f"tagged_{Path(a.input).stem}.md")
    storage.write_tagged_document(tagged, out, Path(a.input).name)
    for t in tagged:
        print(f"{t['index']:>3}  {t['label']:<26} {t['confidence']:.2f}  {t['text'][:70]}")
    print(f"\ntagged {len(tagged)} clauses -> {out}")


# -------------------------------------------------------------------- sweep --

def cmd_sweep(a):
    rows = []
    for r in a.ranks:
        out = Path(a.out) / f"r{r}"
        print(f"\n=== rank {r} ===")
        train_run(a.data, str(out), rank=r, epochs=a.epochs, tiny=a.tiny, model_name=a.model,
                  batch_size=a.batch_size, lr=a.lr, emb_lr=a.emb_lr, targets=tuple(a.targets), device=a.device)
        m = evaluate_run(str(out), device=a.device, latency_n=20)
        rows.append((f"LoRA r={r}", m))
    if a.with_full:
        out = Path(a.out) / "full"
        print("\n=== full fine-tune baseline ===")
        train_run(a.data, str(out), epochs=a.epochs, tiny=a.tiny, full=True, model_name=a.model, lr=a.lr,
                  batch_size=a.batch_size, device=a.device)
        rows.append(("Full FT", evaluate_run(str(out), device=a.device, latency_n=20)))
    lines = ["| Setup | Accuracy | Macro F1 | Trainable params | % | Checkpoint MB | s/step |", "|---|---|---|---|---|---|---|"]
    for name, m in rows:
        t, e = m["tagging"], m["efficiency"]
        lines.append(f"| {name} | {t['accuracy']:.4f} | {t['macro_f1']:.4f} | {e['trainable_params']:,} | "
                     f"{e['trainable_pct']:.3f} | {e['adapter_mb']:.2f} | {e['sec_per_step']:.3f} |")
    table = "\n".join(lines) + "\n"
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "sweep.md").write_text("# LoRA rank sweep\n\n" + table, encoding="utf-8")
    print("\n" + table)


# --------------------------------------------------------------------- main --

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lora-tagger", description="Tag legal clauses with GPT-2 + LoRA")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("hello", help="check the install works")
    s.set_defaults(fn=cmd_hello)

    s = sub.add_parser("prepare", help="build the clause dataset")
    s.add_argument("--source", default="synthetic", help="synthetic | ledgar | path/to/file.csv|jsonl")
    s.add_argument("--out", default="data/clauses.jsonl")
    s.add_argument("--per-label", type=int, default=60)
    s.add_argument("--top-k", type=int, default=10)
    s.add_argument("--seed", type=int, default=13)
    s.set_defaults(fn=cmd_prepare)

    def common_train(s):
        s.add_argument("--data", default="data/clauses.jsonl")
        s.add_argument("--model", default=None, help="HF model id, default $MODEL_NAME or gpt2")
        s.add_argument("--epochs", type=int, default=3)
        s.add_argument("--batch-size", type=int, default=8)
        s.add_argument("--targets", nargs="+", default=["c_attn"])
        s.add_argument("--tiny", action="store_true", help="small random GPT-2, no download")
        s.add_argument("--device", default=None)

    s = sub.add_parser("train", help="fine-tune with LoRA")
    common_train(s)
    s.add_argument("--out", default="runs/r8")
    s.add_argument("--rank", type=int, default=8)
    s.add_argument("--alpha", type=float, default=None, help="default 2*rank")
    s.add_argument("--lr", type=float, default=2e-4, help="LoRA learning rate")
    s.add_argument("--emb-lr", type=float, default=5e-3, help="learning rate for the new tag-token embeddings")
    s.add_argument("--max-len", type=int, default=256)
    s.add_argument("--full", action="store_true", help="full fine-tune baseline instead of LoRA")
    s.add_argument("--seed", type=int, default=13)
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("evaluate", help="accuracy + efficiency on the held-out split")
    s.add_argument("--run", default="runs/r8")
    s.add_argument("--device", default=None)
    s.set_defaults(fn=cmd_evaluate)

    s = sub.add_parser("merge", help="fold LoRA into GPT-2 and save a plain model")
    s.add_argument("--run", default="runs/r8")
    s.set_defaults(fn=cmd_merge)

    s = sub.add_parser("tag", help="tag every clause in a contract text file")
    s.add_argument("--run", default="runs/r8")
    s.add_argument("--input", required=True)
    s.add_argument("--out", default=None)
    s.add_argument("--device", default=None)
    s.set_defaults(fn=cmd_tag)

    s = sub.add_parser("sweep", help="compare LoRA ranks (and optionally full FT)")
    common_train(s)
    s.add_argument("--ranks", type=int, nargs="+", default=[2, 4, 8, 16])
    s.add_argument("--out", default="runs/sweep")
    s.add_argument("--with-full", action="store_true")
    s.add_argument("--lr", type=float, default=2e-4)
    s.add_argument("--emb-lr", type=float, default=5e-3)
    s.set_defaults(fn=cmd_sweep)
    return p


def main(argv=None):
    _load_env()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    args = build_parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
