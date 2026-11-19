"""Milestone 2: load GPT-2 and extend its tokenizer with our own tag tokens.

Each clause becomes a tiny causal-LM example:

    <|clause|> The parties agree that ... <|tag|> <|tag:governing_law|> <|endoftext|>

`<|clause|>` and `<|tag|>` are structural markers. Every label gets its own
single token (`<|tag:governing_law|>`), so predicting a tag means predicting
one token. That keeps decoding cheap and lets us score all tags in one pass.
"""
from __future__ import annotations

import os

import torch

CLAUSE_TOKEN = "<|clause|>"
TAG_TOKEN = "<|tag|>"
EOS = "<|endoftext|>"


def label_token(label: str) -> str:
    return f"<|tag:{label}|>"


def get_device(pref: str | None = None) -> torch.device:
    pref = (pref or os.getenv("DEVICE", "auto")).lower()
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def extend_tokenizer(tokenizer, labels: list[str]) -> dict:
    """Add the marker + label tokens. Returns info needed to resize the model."""
    old_size = len(tokenizer)
    new_tokens = [CLAUSE_TOKEN, TAG_TOKEN] + [label_token(l) for l in labels]
    tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # GPT-2 ships without a pad token
    return {
        "old_vocab_size": old_size,
        "new_vocab_size": len(tokenizer),
        "label_token_ids": [tokenizer.convert_tokens_to_ids(label_token(l)) for l in labels],
    }


def resize_embeddings(model, tokenizer, old_vocab_size: int) -> None:
    """Grow wte (tied with lm_head) and start new rows at the mean embedding.

    Random init for new rows makes the model wildly over-predict them at step 0;
    mean init keeps the starting loss sane, which matters with few training steps.
    """
    try:
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    except TypeError:  # transformers < 4.46
        model.resize_token_embeddings(len(tokenizer))
    with torch.no_grad():
        wte = model.get_input_embeddings().weight
        wte[old_vocab_size:] = wte[:old_vocab_size].mean(dim=0, keepdim=True)


def load_base(model_name: str | None = None, labels: list[str] | None = None):
    """Real GPT-2 from the Hugging Face hub (needs internet the first time)."""
    from transformers import AutoTokenizer, GPT2LMHeadModel

    model_name = model_name or os.getenv("MODEL_NAME", "gpt2")
    tok = AutoTokenizer.from_pretrained(model_name, token=os.getenv("HF_TOKEN") or None)
    model = GPT2LMHeadModel.from_pretrained(model_name, token=os.getenv("HF_TOKEN") or None)
    info = extend_tokenizer(tok, labels or [])
    resize_embeddings(model, tok, info["old_vocab_size"])
    return tok, model, info


def _pretrain_tiny(model, tok, corpus: list[str], steps: int, seed: int) -> None:
    """A few hundred steps of plain next-token prediction on the corpus.

    LoRA assumes a *pretrained* base: it nudges features that already exist.
    A random base has nothing to nudge (we tried, it sits at chance), so the
    tiny stand-in gets a quick pretraining pass first. Real GPT-2 skips this."""
    import random
    rng = random.Random(seed)
    enc = [tok(t, add_special_tokens=False)["input_ids"][:96] + [tok.eos_token_id] for t in corpus if t]
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    model.train()
    for _ in range(steps):
        batch = rng.sample(enc, min(16, len(enc)))
        n = max(map(len, batch))
        ids = torch.full((len(batch), n), tok.eos_token_id)
        lab = torch.full((len(batch), n), -100)
        att = torch.zeros((len(batch), n), dtype=torch.long)
        for i, e in enumerate(batch):
            ids[i, :len(e)] = torch.tensor(e)
            lab[i, :len(e)] = torch.tensor(e)
            att[i, :len(e)] = 1
        model(input_ids=ids, attention_mask=att, labels=lab).loss.backward()
        opt.step()
        opt.zero_grad()
    model.eval()


def load_tiny(corpus: list[str], labels: list[str], seed: int = 0, pretrain_steps: int = 300):
    """A small GPT-2 plus a BPE tokenizer trained on the corpus, briefly
    pretrained as a language model. No downloads, so tests and CI can run the
    whole pipeline offline. Same architecture class as real GPT-2, just shrunk."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    raw = Tokenizer(models.BPE())
    raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    raw.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=1000, special_tokens=[EOS],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    raw.train_from_iterator(corpus, trainer)
    tok = PreTrainedTokenizerFast(tokenizer_object=raw, eos_token=EOS, bos_token=EOS, unk_token=EOS)

    torch.manual_seed(seed)
    cfg = GPT2Config(vocab_size=len(tok), n_positions=256, n_embd=64, n_layer=2, n_head=4,
                     bos_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
    model = GPT2LMHeadModel(cfg)
    if pretrain_steps and corpus:
        _pretrain_tiny(model, tok, corpus, pretrain_steps, seed)
    info = extend_tokenizer(tok, labels)
    resize_embeddings(model, tok, info["old_vocab_size"])
    return tok, model, info


def build_model(labels: list[str], model_name: str | None = None, tiny: bool = False,
                corpus: list[str] | None = None):
    if tiny:
        return load_tiny(corpus or [], labels)
    return load_base(model_name, labels)
