# Legal Document LoRA Tagger

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/qy8w4fe9g8/badge.svg)](https://ask.thili.ai/b/qy8w4fe9g8)

This project develops a Legal Document LoRA Tagger to help public sector clerks classify documents faster. By fine-tuning GPT-2 with a LoRA adapter, the tool efficiently tags legal clauses in public records, reducing manual review time and improving operational efficiency. The project demonstrates the application of LoRA for efficient fine-tuning and GPT-2's capability for specialized text classification.

## What you are building

**Goal.** You will build a tool that fine-tunes GPT-2 using LoRA to automatically tag legal clauses in public records, significantly improving the efficiency of document classification.

**Done when.** You will have successfully fine-tuned GPT-2 with a LoRA adapter to classify legal contract clauses, resulting in a tagged legal document with evaluated accuracy and efficiency.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Prepare a dataset of legal clauses for tagging**  
      Hint: Load and preprocess your legal clause data for training, ensuring it's in a suitable format for sequence tagging.
- [ ] **2. Extend GPT-2 tokenizer and model for custom tags**  
      Hint: Extend the GPT-2 tokenizer with new tokens for your custom tags and resize the model's token embeddings accordingly, as shown in `lfllm-lflegalgpt`.
- [ ] **3. Implement LoRA adapter and inject into GPT-2**  
      Hint: Implement the LoRA adapter technique from `Build LoRA from Scratch` and inject it into the attention layers of your GPT-2 model.
- [ ] **4. Fine-tune GPT-2 with LoRA on the clause dataset**  
      Hint: Apply the fine-tuning technique from `Fine-Tune GPT-2 to Tag Legal Contract Clauses`, utilizing your LoRA-adapted GPT-2 model on the prepared clause dataset.
- [ ] **5. Merge LoRA adapter and perform inference on new clauses**  
      Hint: Merge the LoRA adapter weights back into the base GPT-2 model using the technique from `Build LoRA from Scratch`, then perform inference on new, unseen legal clauses.
- [ ] **6. Evaluate tagging accuracy and model efficiency**  
      Hint: Calculate relevant metrics to assess the accuracy of your clause tagging and analyze the efficiency gains from using LoRA.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Analyze the model's performance on specific types of legal clauses to identify areas for improvement.
- Experiment with different LoRA configurations (e.g., rank, alpha) to observe their impact on tagging accuracy and training efficiency.
- Explore additional metrics for evaluating model efficiency, such as inference speed or memory footprint, beyond just tagging accuracy.
