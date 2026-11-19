# legal-document-lora-tagger

Fine-tunes GPT-2 with a LoRA adapter (written from scratch, no `peft`) to tag legal contract clauses: `governing_law`, `termination`, `indemnification`, and so on. Give it a contract as plain text and you get back a tagged document with a confidence score for every clause.

## How it works

Every clause becomes a short causal-LM example:

```
<|clause|> This Agreement shall be governed by the laws of Karnataka. <|tag|> <|tag:governing_law|> <|endoftext|>
```

- **Tokenizer extension.** Each tag is one new token, and the embedding matrix is resized to fit. New rows start at the mean of the existing embeddings, not random noise.
- **LoRA.** `LoRALayer` wraps GPT-2's fused q/k/v projection (`attn.c_attn`) as `h = xW + (α/r)·xAB`, with `B` set to zero at init. The whole base model is frozen. The only things that train are `A`, `B` and the new tag-token rows (a gradient hook zeroes updates to the original 50,257 rows).
- **Training.** Loss is masked so only the tag token and EOS are supervised.
- **Inference.** The model reads the prompt once, and the answer is the most likely of the tag tokens. It can't invent a tag that isn't on the list, and the softmax over tags gives you a confidence score.
- **Merge.** `W ← W + (α/r)·AB`, after which the LoRA wrapper is gone. Evaluation checks that predictions match before and after the merge.

## Code map

| Milestone | Where |
|---|---|
| 1. Dataset | `app/dataset.py`: synthetic templates, LEDGAR loader, your own CSV/JSONL, stratified split, contract splitter |
| 2. Tokenizer + embeddings | `app/llm.py`: `extend_tokenizer`, `resize_embeddings` |
| 3. LoRA | `app/core.py`: `LoRALayer`, `inject_lora`, `train_new_token_rows` |
| 4. Fine-tune | `app/core.py`: `encode_example`, `train` |
| 5. Merge + inference | `app/core.py`: `merge_lora`, `predict`, `tag_document`; `app/storage.py` for save/load |
| 6. Evaluation | `app/core.py`: `classification_report`, `time_inference`; `app/cli.py`: `evaluate_run` writes `report.md` |

## Quick start

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python -m app.cli hello      # checks the install

# 1. data: real contract provisions from LEDGAR, top 10 tags, 300 each
python -m app.cli prepare --source ledgar --top-k 10 --per-label 300

# 2-4. fine-tune GPT-2 with LoRA r=8
python -m app.cli train --rank 8 --epochs 3 --out runs/r8

# 5. merge the adapter into GPT-2, then tag a contract it has never seen
python -m app.cli merge --run runs/r8
python -m app.cli tag --run runs/r8 --input examples/sample_contract.txt

# 6. accuracy, per-tag F1, confusion matrix, efficiency -> runs/r8/report.md
python -m app.cli evaluate --run runs/r8
```

Note: the sample contract uses the synthetic tag names. With LEDGAR, the tags are LEDGAR's own labels, so pass any contract text you like.

On a GPU this takes a few minutes. On a laptop CPU, budget around half an hour for 3 epochs. Google Colab's free T4 works fine.

**No internet / just checking it works:** add `--tiny` to `train` or `sweep`. That swaps in a 2-layer GPT-2 with a tokenizer trained on the spot and a quick LM pretraining pass. Everything runs offline in about 20 seconds, which is also what CI uses.

```bash
python -m app.cli prepare --source synthetic
python -m app.cli train --tiny --epochs 8 --lr 3e-3 --emb-lr 1e-2 --batch-size 16 --out runs/tiny-r8
```

## Results so far (tiny model, synthetic data)

Test split: 90 held-out clauses, 10 tags.

| Setup | Accuracy | Macro F1 | Trainable params | % of model | Checkpoint |
|---|---|---|---|---|---|
| LoRA r=1 | 0.767 | 0.764 | 1,280 | 0.70% | 0.01 MB |
| LoRA r=4 | 0.900 | 0.899 | 2,816 | 1.54% | 0.01 MB |
| LoRA r=8 | 0.956 | 0.955 | 4,864 | 2.62% | 0.02 MB |
| LoRA r=16 | 0.967 | 0.966 | 8,960 | 4.73% | 0.04 MB |
| Full fine-tune | 1.000 | 1.000 | 181,248 | 100% | 0.69 MB |

The merged model gave 100% identical predictions, with a max probability drift of 7e-7.

Honest caveats:
- The synthetic set is template-generated, so these numbers are optimistic.
- On the hand-written `examples/sample_contract.txt` the tiny model got 4 of 10 right. It learned template vocabulary, not legal meaning.
- The real test is GPT-2 on LEDGAR. Run that and replace this table with its numbers.

For the full-size model, the efficiency story is what you'd expect. LoRA r=8 on `c_attn` trains about 0.3M params out of 124M (roughly 0.24%). The adapter is about 1 MB against roughly 475 MB for a full checkpoint, and gradient + AdamW state shrinks by about 400×.

## Things I learned building it

- **LoRA on a random base doesn't learn.** It sat at chance (10%) because there were no useful features for the low-rank update to steer. Hence the pretraining pass for `--tiny`.
- **The new tag embeddings need their own, higher learning rate.** At 1e-3, val accuracy stayed flat at chance for 8 epochs. At 1e-2 it reached 95%. That's why `--emb-lr` exists.

## Stretch experiments

```bash
# rank sweep + full fine-tune baseline -> runs/sweep/sweep.md
python -m app.cli sweep --ranks 2 4 8 16 --with-full

# adapt the attention output projection too
python -m app.cli train --targets c_attn c_proj --out runs/r8-qkvo

# your own data (columns: text, label)
python -m app.cli prepare --source my_clauses.csv
```

Each `report.md` lists per-tag F1 sorted worst-first, a confusion matrix and sample mistakes. That's the place to look for clause types that need more data.

## Tests

```bash
pytest -m "not slow"    # tiny model, no downloads
pytest -m slow          # also loads real GPT-2
```

They cover LoRA being an identity at init, merge being exact for both `Conv1D` and `nn.Linear`, the base staying frozen, original embedding rows never moving, loss masking, the metrics maths, and a full prepare → train → evaluate → merge → tag run.
