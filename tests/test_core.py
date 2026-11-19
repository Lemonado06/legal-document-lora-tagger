"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

All six milestones are done, so none of these are skipped any more.
They run on a tiny GPT-2 (no downloads) so they pass on GitHub Actions.
"""
import copy

import pytest
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.pytorch_utils import Conv1D

from app import core, dataset, llm

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def tiny():
    rows = dataset.make_synthetic(n_per_label=8, seed=1)
    labels = sorted({r["label"] for r in rows})
    tok, model, info = llm.load_tiny([r["text"] for r in rows], labels, pretrain_steps=20)
    label2id = dict(zip(labels, info["label_token_ids"]))
    return rows, labels, tok, model, info, label2id


# Milestone 1: Prepare a dataset of legal clauses for tagging
def test_milestone_1():
    rows = dataset.make_synthetic(n_per_label=20, seed=3)
    # every tag present, no duplicate clauses, text cleaned
    assert {r["label"] for r in rows} == set(dataset.LABELS)
    assert len({r["text"] for r in rows}) == len(rows)
    assert all(r["text"] == dataset.clean_text(r["text"]) for r in rows)

    # stratified split: every tag reaches the test set, nothing leaks between splits
    tr, va, te = dataset.stratified_split(rows)
    assert {r["label"] for r in te} == set(dataset.LABELS)
    assert not ({r["text"] for r in tr} & {r["text"] for r in te})
    assert len(tr) + len(va) + len(te) == len(rows)

    # contracts split into clauses, headings dropped, numbering stripped
    doc = "MASTER AGREEMENT\n\n1. First clause about payment of invoices.\n2. Second clause about governing law here."
    assert dataset.split_document(doc) == ["First clause about payment of invoices.",
                                           "Second clause about governing law here."]


# Milestone 2: Extend GPT-2 tokenizer and model for custom tags
def test_milestone_2(tiny):
    rows, labels, tok, model, info, label2id = tiny
    # 2 markers + one token per tag, embeddings resized to match
    assert info["new_vocab_size"] == info["old_vocab_size"] + 2 + len(labels)
    assert model.get_input_embeddings().weight.shape[0] == len(tok)
    # each tag is a single token, not split into pieces
    for lab in labels:
        assert len(tok(llm.label_token(lab), add_special_tokens=False)["input_ids"]) == 1
    # new rows start at the mean of the old ones
    wte = model.get_input_embeddings().weight
    old = info["old_vocab_size"]
    assert torch.allclose(wte[old], wte[:old].mean(0), atol=1e-5)


# Milestone 3: Implement LoRA adapter and inject into GPT-2
@pytest.mark.parametrize("base", [Conv1D(12, 6), nn.Linear(6, 12)])
def test_milestone_3(base):
    torch.manual_seed(0)
    x = torch.randn(3, 5, 6)
    ref = base(x).detach()
    layer = core.LoRALayer(base, r=4, alpha=8)
    assert torch.allclose(layer(x), ref)  # B starts at zero -> no change at init
    assert not any(p.requires_grad for p in layer.base.parameters())

    # injection hits attention only and freezes everything else
    m = GPT2LMHeadModel(GPT2Config(vocab_size=100, n_positions=32, n_embd=32, n_layer=3, n_head=2))
    names = core.inject_lora(m, targets=("c_attn", "c_proj"), r=2)
    assert len(names) == 6 and all(".attn." in n for n in names)  # mlp.c_proj untouched
    trainable = [n for n, p in m.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n in trainable)


# Milestone 4: Fine-tune GPT-2 with LoRA on the clause dataset
def test_milestone_4(tiny):
    rows, labels, tok, model, info, label2id = tiny
    # loss only on the tag token (+ EOS), prompt masked out
    lid = info["label_token_ids"][0]
    ex = core.encode_example(tok, "Some clause text.", lid, max_len=64)
    assert [t for t in ex["labels"] if t != -100] == [lid, tok.eos_token_id]
    long = core.encode_example(tok, "word " * 500, lid, max_len=64)
    assert len(long["input_ids"]) == 64 and long["labels"][-2] == lid

    # a short run: loss goes down, LoRA + new rows move, pretrained rows don't
    m = copy.deepcopy(model)
    core.inject_lora(m, r=4)
    core.train_new_token_rows(m, info["old_vocab_size"])
    old = info["old_vocab_size"]
    before = m.get_input_embeddings().weight.detach().clone()
    cfg = core.TrainConfig(epochs=3, batch_size=8, lr=3e-3, emb_lr=1e-2, log_every=10**9)
    res = core.train(m, tok, rows, [], label2id, cfg, CPU, log=lambda *_: None)
    after = m.get_input_embeddings().weight.detach()
    assert res.history[-1]["train_loss"] < res.history[0]["train_loss"]
    assert torch.equal(before[:old], after[:old])
    assert not torch.equal(before[old:], after[old:])
    assert core.count_params(m, old)["trainable_pct"] < 50


# Milestone 5: Merge LoRA adapter and perform inference on new clauses
def test_milestone_5(tiny):
    rows, labels, tok, model, info, label2id = tiny
    m = copy.deepcopy(model)
    core.inject_lora(m, r=4)
    torch.manual_seed(1)
    with torch.no_grad():
        for layer in core.lora_layers(m).values():
            layer.lora_B.normal_(std=0.05)
    texts = [r["text"] for r in rows[:10]]
    before = core.predict(m, tok, texts, label2id, CPU)

    assert core.merge_lora(m) == 2
    assert not core.lora_layers(m)  # wrappers gone, plain GPT-2 again
    after = core.predict(m, tok, texts, label2id, CPU)
    for a, b in zip(before, after):
        assert a["label"] == b["label"]
        assert abs(a["confidence"] - b["confidence"]) < 1e-4

    # inference on unseen clauses only ever returns known tags
    tagged = core.tag_document(m, tok, ["The courts of Mumbai have jurisdiction.", "hello"], label2id, CPU)
    assert [t["index"] for t in tagged] == [1, 2]
    assert all(t["label"] in labels and 0 < t["confidence"] <= 1 for t in tagged)


# Milestone 6: Evaluate tagging accuracy and model efficiency
def test_milestone_6(tiny):
    y = ["a", "a", "b", "b", "c"]
    p = ["a", "b", "b", "b", "a"]
    r = core.classification_report(y, p, ["a", "b", "c"])
    assert r["accuracy"] == pytest.approx(0.6)
    assert r["per_label"]["a"]["precision"] == pytest.approx(0.5)
    assert r["per_label"]["b"]["recall"] == pytest.approx(1.0)
    assert r["per_label"]["c"]["f1"] == 0
    assert r["confusion"]["a"]["b"] == 1

    # efficiency numbers are sane
    rows, labels, tok, model, info, label2id = tiny
    m = copy.deepcopy(model)
    core.inject_lora(m, r=2)
    core.train_new_token_rows(m, info["old_vocab_size"])
    params = core.count_params(m, info["old_vocab_size"])
    assert params["trainable"] == params["lora"] + params["new_embedding_rows"]
    assert params["new_embedding_rows"] == (len(tok) - info["old_vocab_size"]) * m.config.n_embd
    assert core.time_inference(m, tok, [r["text"] for r in rows[:3]], label2id, CPU) > 0
