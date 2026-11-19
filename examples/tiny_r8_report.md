# Evaluation: LoRA r=8, alpha=16, targets=['c_attn']

Base: `tiny-gpt2 (random init)`, data: `data/clauses.jsonl`

## Tagging

- Accuracy: **0.9556** on 90 held-out clauses
- Macro F1: **0.9550**, weighted F1: 0.9550

| Tag | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| `dispute_resolution` | 1.000 | 0.778 | 0.875 | 9 |
| `limitation_of_liability` | 1.000 | 0.778 | 0.875 | 9 |
| `governing_law` | 0.818 | 1.000 | 0.900 | 9 |
| `payment` | 0.818 | 1.000 | 0.900 | 9 |
| `assignment` | 1.000 | 1.000 | 1.000 | 9 |
| `confidentiality` | 1.000 | 1.000 | 1.000 | 9 |
| `force_majeure` | 1.000 | 1.000 | 1.000 | 9 |
| `indemnification` | 1.000 | 1.000 | 1.000 | 9 |
| `termination` | 1.000 | 1.000 | 1.000 | 9 |
| `warranties` | 1.000 | 1.000 | 1.000 | 9 |

### Confusion matrix (rows = true, cols = predicted)

| | assignment | confidenti | dispute_re | force_maje | governing_ | indemnific | limitation | payment | terminatio | warranties |
|---|---|---|---|---|---|---|---|---|---|---|
| assignment | 9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| confidenti | 0 | 9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| dispute_re | 0 | 0 | 7 | 0 | 2 | 0 | 0 | 0 | 0 | 0 |
| force_maje | 0 | 0 | 0 | 9 | 0 | 0 | 0 | 0 | 0 | 0 |
| governing_ | 0 | 0 | 0 | 0 | 9 | 0 | 0 | 0 | 0 | 0 |
| indemnific | 0 | 0 | 0 | 0 | 0 | 9 | 0 | 0 | 0 | 0 |
| limitation | 0 | 0 | 0 | 0 | 0 | 0 | 7 | 2 | 0 | 0 |
| payment | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 9 | 0 | 0 |
| terminatio | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 9 | 0 |
| warranties | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 9 |

## Efficiency

- Trainable params: 4,864 of 185,344 (**2.624%**)
- Adapter on disk: 0.02 MB vs 0.7 MB for a full fp32 checkpoint (~34x smaller)
- Grad + AdamW state: ~0.1 MB with LoRA vs ~2.1 MB for full fine-tuning
- Training: 8.0s total, 35 ms/step, peak memory 881 MB
- Inference (batch 1): 2.14 ms/clause with adapter, 2.14 ms/clause after merge
- Merge check: 100.0% identical predictions, max probability drift 7.15e-07

## Some mistakes

- gold `dispute_resolution`, predicted `governing_law` (0.594): Save as otherwise provided, the courts of London shall have exclusive jurisdiction over any dispute arising under this Agreement. The parties acknowledge that t
- gold `dispute_resolution`, predicted `governing_law` (0.667): For the avoidance of doubt, the courts of Bengaluru shall have exclusive jurisdiction over any dispute arising under this Agreement.
- gold `limitation_of_liability`, predicted `payment` (0.846): Save as otherwise provided, the aggregate liability of the Disclosing Party under this Agreement shall not exceed the total fees paid in the twelve (12) months 
- gold `limitation_of_liability`, predicted `payment` (0.728): Subject to the terms hereof, the aggregate liability of the Vendor under this Agreement shall not exceed the total fees paid in the twelve (12) months preceding
