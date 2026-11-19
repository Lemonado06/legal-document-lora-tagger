"""Saving and loading runs.

A run folder looks like:
    runs/r8/
      config.json        labels, LoRA settings, base model name, data splits used
      adapter.pt         LoRA A/B matrices + the new tag-token embedding rows only
      tokenizer/         the extended tokenizer
      tiny_base.pt       (only for --tiny runs, since that base is random)
      merged/            (after `merge`) full GPT-2 with LoRA folded in
      metrics.json, report.md, history.json
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from . import core, llm


def save_json(obj, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def adapter_state(model, first_new_id: int) -> dict:
    state = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if "lora_" in n}
    state["__new_embedding_rows__"] = model.get_input_embeddings().weight[first_new_id:].detach().cpu().clone()
    return state


def save_run(run_dir: str | Path, model, tokenizer, config: dict) -> dict:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(run_dir / "tokenizer")
    save_json(config, run_dir / "config.json")
    if config.get("full_finetune"):
        model.save_pretrained(run_dir / "full")
        return {"adapter_bytes": dir_size(run_dir / "full")}
    torch.save(adapter_state(model, config["old_vocab_size"]), run_dir / "adapter.pt")
    return {"adapter_bytes": (run_dir / "adapter.pt").stat().st_size}


def save_tiny_base(run_dir: str | Path, model) -> None:
    """Snapshot the random tiny base BEFORE LoRA so reloads start identical."""
    torch.save({"config": model.config.to_dict(), "state": model.state_dict()}, Path(run_dir) / "tiny_base.pt")


def dir_size(path: str | Path) -> int:
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())


def load_run(run_dir: str | Path, merged: bool = False):
    """Rebuild base -> extend tokenizer -> inject LoRA -> load adapter.
    With merged=True and a merged/ folder present, load that directly."""
    from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

    run_dir = Path(run_dir)
    cfg = load_json(run_dir / "config.json")
    tok = AutoTokenizer.from_pretrained(run_dir / "tokenizer")
    label2id = {l: tok.convert_tokens_to_ids(llm.label_token(l)) for l in cfg["labels"]}

    if merged and (run_dir / "merged").exists():
        return tok, GPT2LMHeadModel.from_pretrained(run_dir / "merged"), cfg, label2id
    if cfg.get("full_finetune"):
        return tok, GPT2LMHeadModel.from_pretrained(run_dir / "full"), cfg, label2id

    if cfg.get("tiny"):
        snap = torch.load(run_dir / "tiny_base.pt", weights_only=False)
        model = GPT2LMHeadModel(GPT2Config.from_dict(snap["config"]))
        model.load_state_dict(snap["state"])
    else:
        model = GPT2LMHeadModel.from_pretrained(cfg["base_model"])
        model.resize_token_embeddings(len(tok))

    core.inject_lora(model, targets=tuple(cfg["targets"]), r=cfg["rank"], alpha=cfg["alpha"], dropout=0.0)
    state = torch.load(run_dir / "adapter.pt", weights_only=True)
    rows = state.pop("__new_embedding_rows__")
    with torch.no_grad():
        model.get_input_embeddings().weight[cfg["old_vocab_size"]:] = rows
    missing = [n for n, _ in model.named_parameters() if "lora_" in n and n not in state]
    if missing:
        raise RuntimeError(f"adapter is missing {len(missing)} LoRA tensors, e.g. {missing[0]}")
    params = dict(model.named_parameters())
    with torch.no_grad():
        for n, t in state.items():
            params[n].copy_(t)
    return tok, model, cfg, label2id


def write_tagged_document(tagged: list[dict], out: str | Path, source_name: str = "document") -> Path:
    """Markdown for people, JSON next to it for code."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# Tagged clauses: {source_name}", "", "| # | Tag | Confidence | Clause |", "|---|---|---|---|"]
    for t in tagged:
        clause = t["text"].replace("|", "\\|")
        if len(clause) > 220:
            clause = clause[:217] + "..."
        lines.append(f"| {t['index']} | `{t['label']}` | {t['confidence']:.2f} | {clause} |")
    counts: dict[str, int] = {}
    for t in tagged:
        counts[t["label"]] = counts.get(t["label"], 0) + 1
    lines += ["", "## Tag counts", ""] + [f"- `{k}`: {v}" for k, v in sorted(counts.items(), key=lambda x: -x[1])]
    low = [t for t in tagged if t["confidence"] < 0.6]
    if low:
        lines += ["", "## Worth a human look (confidence < 0.60)", ""] + [f"- #{t['index']} `{t['label']}`" for t in low]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    save_json(tagged, out.with_suffix(".json"))
    return out
