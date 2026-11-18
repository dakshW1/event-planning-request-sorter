# Event Planning Request Sorter

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/dkw29pbvgc/badge.svg)](https://ask.thili.ai/b/dkw29pbvgc)

This tool helps event planners automatically sort incoming client requests like "venue inquiry" or "catering request" for efficient task delegation and response prioritization. It leverages a fine-tuned GPT-2 model to classify short textual requests by generating special label tokens, demonstrating advanced natural language processing capabilities.

## What you are building

**Goal.** You will build an Event Planning Request Sorter that automatically classifies incoming client requests (e.g., "venue inquiry", "catering request") to help event planners efficiently delegate tasks and prioritize responses. This project will demonstrate the fine-tuning logic from the GPT-2 drill to classify short textual requests and implement an inference pipeline for predictions.

**Done when.** You will have defined common event request types and created a small dataset. You will have configured and trained a GPT-2 model to classify these requests. Finally, you will have built a command-line interface (CLI) that takes new requests, displays their classification, and allows saving them to a file.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Define common event request types and create a small dataset of example requests.**  
      Hint: Identify common request types and create a small dataset of example requests, similar to how legal clauses were identified.
- [ ] **2. Configure GPT-2 with an extended tokenizer and model for custom tokens.**  
      Hint: Extend the GPT-2 tokenizer and configure the model for custom tokens, as shown in the drill for special tags.
- [ ] **3. Train the model on the example requests using the Hugging Face Trainer.**  
      Hint: Use the Hugging Face Trainer to fine-tune your GPT-2 model on your dataset of event requests.
- [ ] **4. Build an interface (CLI) to input new requests and show their classification.**  
      Hint: Implement the inference pipeline to take new requests and predict their classification, mirroring the prediction step in the drill.
- [ ] **5. Allow saving classified requests to a file for review.**  
      Hint: Add functionality to your CLI to save the input request and its predicted classification to a file.

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

- (none)
