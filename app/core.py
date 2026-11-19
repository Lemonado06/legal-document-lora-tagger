"""The interesting part: LoRA from scratch, fine-tuning, merging, inference, metrics.

Milestone 3  LoRALayer / inject_lora
Milestone 4  encode_example / train
Milestone 5  merge_lora / predict / tag_document
Milestone 6  classification_report / efficiency helpers
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from .llm import CLAUSE_TOKEN, TAG_TOKEN

# ======================================================================
# Milestone 3: LoRA
# ======================================================================


class LoRALayer(nn.Module):
    """Wraps a frozen projection W and adds a low-rank update:

        h = x W + (alpha / r) * x A B

    A is (in, r), B is (r, out). B starts at zero, so at step 0 the wrapped
    layer gives exactly the same output as the original one.

    GPT-2 uses `Conv1D` (weight stored as (in, out)), most other models use
    `nn.Linear` (weight stored as (out, in)). Both are handled.
    """

    def __init__(self, base: nn.Module, r: int = 8, alpha: float = 16, dropout: float = 0.0):
        super().__init__()
        if r <= 0:
            raise ValueError("rank must be > 0")
        self.base = base
        self.is_conv1d = isinstance(base, Conv1D)
        if self.is_conv1d:
            in_f, out_f = base.weight.shape
        elif isinstance(base, nn.Linear):
            out_f, in_f = base.weight.shape
        else:
            raise TypeError(f"can't wrap {type(base).__name__}")
        self.r, self.alpha, self.scaling = r, alpha, alpha / r
        self.lora_A = nn.Parameter(torch.empty(in_f, r, dtype=base.weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros(r, out_f, dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.merged = False
        for p in self.base.parameters():
            p.requires_grad = False

    def delta_weight(self) -> torch.Tensor:
        """The update in the base layer's own weight layout."""
        d = (self.lora_A @ self.lora_B) * self.scaling  # (in, out)
        return d if self.is_conv1d else d.T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if not self.merged:
            out = out + (self.dropout(x) @ self.lora_A @ self.lora_B) * self.scaling
        return out

    @torch.no_grad()
    def merge(self) -> nn.Module:
        """Fold A B into W and return the plain base layer (no extra matmuls)."""
        if not self.merged:
            self.base.weight += self.delta_weight().to(self.base.weight.dtype)
            self.merged = True
        return self.base


def _set_child(root: nn.Module, dotted: str, new: nn.Module) -> None:
    parent_name, _, child = dotted.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child, new)


def inject_lora(model: nn.Module, targets=("c_attn",), r: int = 8, alpha: float = 16,
                dropout: float = 0.05) -> list[str]:
    """Freeze the model and swap every module whose name ends in one of
    `targets` for a LoRALayer. Default hits GPT-2's fused q/k/v projection.
    Pass targets=("c_attn", "c_proj") to also adapt the attention output."""
    for p in model.parameters():
        p.requires_grad = False
    replaced = []
    for name, module in list(model.named_modules()):
        leaf = name.rsplit(".", 1)[-1]
        if leaf in targets and ".attn." in f".{name}." and isinstance(module, (Conv1D, nn.Linear)):
            _set_child(model, name, LoRALayer(module, r=r, alpha=alpha, dropout=dropout))
            replaced.append(name)
    if not replaced:
        raise RuntimeError(f"no modules matched {targets}")
    return replaced


def lora_layers(model: nn.Module) -> dict[str, LoRALayer]:
    return {n: m for n, m in model.named_modules() if isinstance(m, LoRALayer)}


def merge_lora(model: nn.Module) -> int:
    """Milestone 5: replace each LoRALayer with its merged base layer."""
    layers = lora_layers(model)
    for name, layer in layers.items():
        _set_child(model, name, layer.merge())
    return len(layers)


def train_new_token_rows(model: nn.Module, first_new_id: int) -> None:
    """Our tag tokens are brand new, so their embeddings must be learned.
    Unfreeze wte (tied to lm_head) but zero the gradient for every original
    row, so the 50k pretrained token vectors never move."""
    wte = model.get_input_embeddings().weight
    wte.requires_grad = True

    def _mask(grad):
        grad = grad.clone()
        grad[:first_new_id] = 0
        return grad

    wte.register_hook(_mask)


def count_params(model: nn.Module, first_new_id: int | None = None) -> dict:
    """Trainable count is honest about the embedding trick: only the new rows
    of wte actually learn, so only those are counted."""
    total = sum(p.numel() for p in model.parameters())
    lora = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
    emb = 0
    wte = model.get_input_embeddings().weight
    if wte.requires_grad:
        rows = wte.shape[0] - (first_new_id or 0)
        emb = rows * wte.shape[1]
    other = sum(p.numel() for n, p in model.named_parameters()
                if p.requires_grad and "lora_" not in n and p is not wte)
    trainable = lora + emb + other
    return {"total": total, "trainable": trainable, "lora": lora, "new_embedding_rows": emb,
            "other_trainable": other, "trainable_pct": 100 * trainable / total}


# ======================================================================
# Milestone 4: encoding + training
# ======================================================================


def encode_example(tokenizer, text: str, label_id: int | None, max_len: int = 256) -> dict:
    """Prompt tokens are masked (-100) so loss is only on the tag token + EOS.
    Long clauses are truncated from the middle-ish end: we keep the start of
    the clause because headings and the first sentence carry most signal."""
    clause_id = tokenizer.convert_tokens_to_ids(CLAUSE_TOKEN)
    tag_id = tokenizer.convert_tokens_to_ids(TAG_TOKEN)
    body = tokenizer(" " + text, add_special_tokens=False)["input_ids"]
    room = max_len - 2 - (2 if label_id is not None else 0)
    body = body[:room]
    prompt = [clause_id] + body + [tag_id]
    if label_id is None:
        return {"input_ids": prompt, "labels": [-100] * len(prompt)}
    ids = prompt + [label_id, tokenizer.eos_token_id]
    labels = [-100] * len(prompt) + [label_id, tokenizer.eos_token_id]
    return {"input_ids": ids, "labels": labels}


def collate(batch: list[dict], pad_id: int) -> dict:
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), n), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        ids[i, :L] = torch.tensor(b["input_ids"])
        labels[i, :L] = torch.tensor(b["labels"])
        mask[i, :L] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


@dataclass
class TrainConfig:
    epochs: int = 3
    batch_size: int = 8
    lr: float = 2e-4
    emb_lr: float = 5e-3
    warmup_ratio: float = 0.06
    weight_decay: float = 0.0
    max_len: int = 256
    grad_clip: float = 1.0
    seed: int = 13
    log_every: int = 20
    full_finetune: bool = False


@dataclass
class TrainResult:
    history: list[dict] = field(default_factory=list)
    seconds: float = 0.0
    steps: int = 0
    peak_memory_mb: float | None = None
    sec_per_step: float = 0.0


def _peak_mem_mb(device: torch.device) -> float | None:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 2**20
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux: KB
    except Exception:
        return None


def train(model, tokenizer, train_rows, val_rows, label2id, cfg: TrainConfig, device,
          log=print) -> TrainResult:
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    model.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    data = [encode_example(tokenizer, r["text"], label2id[r["label"]], cfg.max_len) for r in train_rows]
    wte = model.get_input_embeddings().weight
    main = [p for p in model.parameters() if p.requires_grad and p is not wte]
    groups = [{"params": main, "lr": cfg.lr, "weight_decay": cfg.weight_decay}]
    if wte.requires_grad and not cfg.full_finetune:
        groups.append({"params": [wte], "lr": cfg.emb_lr, "weight_decay": 0.0})
    elif wte.requires_grad:
        groups[0]["params"].append(wte)
    opt = torch.optim.AdamW(groups)

    steps_per_epoch = math.ceil(len(data) / cfg.batch_size)
    total = steps_per_epoch * cfg.epochs
    warm = max(1, int(total * cfg.warmup_ratio))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (total - s) / max(1, total - warm)))

    res = TrainResult()
    t0, step = time.perf_counter(), 0
    for epoch in range(cfg.epochs):
        model.train()
        random.shuffle(data)
        running = 0.0
        for i in range(0, len(data), cfg.batch_size):
            batch = {k: v.to(device) for k, v in collate(data[i:i + cfg.batch_size], tokenizer.pad_token_id).items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], cfg.grad_clip)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            running += loss.item()
            if step % cfg.log_every == 0:
                log(f"  epoch {epoch + 1} step {step}/{total} loss {loss.item():.4f}")
        entry = {"epoch": epoch + 1, "train_loss": running / steps_per_epoch}
        if val_rows:
            preds = predict(model, tokenizer, [r["text"] for r in val_rows], label2id, device,
                            max_len=cfg.max_len)
            entry["val_accuracy"] = sum(p["label"] == r["label"] for p, r in zip(preds, val_rows)) / len(val_rows)
        res.history.append(entry)
        log(f"epoch {epoch + 1}: " + ", ".join(f"{k}={v:.4f}" for k, v in entry.items() if k != "epoch"))
    res.seconds = time.perf_counter() - t0
    res.steps = step
    res.sec_per_step = res.seconds / max(1, step)
    res.peak_memory_mb = _peak_mem_mb(device)
    return res


# ======================================================================
# Milestone 5: inference
# ======================================================================


@torch.no_grad()
def predict(model, tokenizer, texts: list[str], label2id: dict, device, batch_size: int = 16,
            max_len: int = 256) -> list[dict]:
    """Constrained decoding: run the prompt once, read the logits at <|tag|>,
    and pick the best of our label tokens only. The model can't answer with
    an off-list tag, and we get a confidence score for free."""
    model.eval()
    labels = list(label2id)
    ids = torch.tensor([label2id[l] for l in labels], device=device)
    out = []
    for i in range(0, len(texts), batch_size):
        enc = [encode_example(tokenizer, t, None, max_len) for t in texts[i:i + batch_size]]
        b = collate(enc, tokenizer.pad_token_id)
        b = {k: v.to(device) for k, v in b.items() if k != "labels"}
        logits = model(**b).logits  # (B, T, V)
        last = b["attention_mask"].sum(1) - 1
        at_tag = logits[torch.arange(logits.size(0), device=device), last]  # (B, V)
        probs = F.softmax(at_tag[:, ids].float(), dim=-1)
        conf, idx = probs.max(-1)
        for j in range(len(enc)):
            out.append({"label": labels[idx[j].item()], "confidence": conf[j].item(),
                        "scores": {labels[k]: probs[j, k].item() for k in range(len(labels))}})
    return out


def tag_document(model, tokenizer, clauses: list[str], label2id, device) -> list[dict]:
    preds = predict(model, tokenizer, clauses, label2id, device)
    return [{"index": i + 1, "text": c, "label": p["label"], "confidence": round(p["confidence"], 4)}
            for i, (c, p) in enumerate(zip(clauses, preds))]


# ======================================================================
# Milestone 6: metrics
# ======================================================================


def classification_report(y_true: list[str], y_pred: list[str], labels: list[str] | None = None) -> dict:
    labels = labels or sorted(set(y_true) | set(y_pred))
    per = {}
    for lab in labels:
        tp = sum(t == lab and p == lab for t, p in zip(y_true, y_pred))
        fp = sum(t != lab and p == lab for t, p in zip(y_true, y_pred))
        fn = sum(t == lab and p != lab for t, p in zip(y_true, y_pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[lab] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}
    with_support = [v for v in per.values() if v["support"]]
    total = sum(v["support"] for v in with_support) or 1
    confusion = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true, y_pred):
        if t in confusion and p in confusion[t]:
            confusion[t][p] += 1
    return {
        "accuracy": sum(t == p for t, p in zip(y_true, y_pred)) / max(1, len(y_true)),
        "macro_f1": sum(v["f1"] for v in with_support) / max(1, len(with_support)),
        "weighted_f1": sum(v["f1"] * v["support"] for v in with_support) / total,
        "per_label": per,
        "confusion": confusion,
        "n": len(y_true),
    }


def time_inference(model, tokenizer, texts, label2id, device, repeats: int = 1) -> float:
    """Milliseconds per clause at batch size 1 (the 'one document at a time' case)."""
    if not texts:
        return 0.0
    predict(model, tokenizer, texts[:2], label2id, device, batch_size=1)  # warm-up
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        predict(model, tokenizer, texts, label2id, device, batch_size=1)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - t0) / (len(texts) * repeats)
