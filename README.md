# Event Planning Request Sorter

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/dkw29pbvgc/badge.svg)](https://ask.thili.ai/b/dkw29pbvgc)
[![tests](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)

Event planners get a mixed inbox: venue questions, menu requests, date changes, cancellations. This CLI sorts each
request into one of 8 types with **GPT-2 fine-tuned the same way as the legal-clause drill**: every type becomes
a special token, and the model learns to write the right token after `Request: ... Type:`. Classified requests can
be saved to a file and reviewed as work queues.

```text
$ python -m app.cli classify "Could you quote us for a 150 guest dinner?" "can we get vegan cupcakes" --save

  Could you quote us for a 150 guest dinner?
  -> budget_quote  (99%)   prices, quotations, packages and discounts
     also: catering_request (1%)

  can we get vegan cupcakes
  -> catering_request  (100%)   food, drinks, menus and dietary needs

$ python -m app.cli review
budget_quote (1)
  [99%] Could you quote us for a 150 guest dinner?
catering_request (1)
  [100%] can we get vegan cupcakes
```

## Milestones

- [x] **1. Define common event request types and create a small dataset of example requests.** 8 types; a
      templated training set (240 requests) and a separate, hand-written test set (40 requests, 5 per type).
- [x] **2. Configure GPT-2 with an extended tokenizer and model for custom tokens.** One special token per type
      plus a pad token, resized embeddings, and a loss on the label token only.
- [x] **3. Train the model on the example requests using the Hugging Face Trainer.**
- [x] **4. Build an interface (CLI) to input new requests and show their classification.** `classify`,
      `interactive`, `evaluate`.
- [x] **5. Allow saving classified requests to a file for review.** `--save` appends JSON lines; `review` groups
      them into one queue per type, plus a `needs_review` queue for low-confidence predictions.

## Run it

You need Python 3.12+ and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python -m app.cli types                      # the 8 request types
python -m app.cli train --lr 1e-4 --epochs 6  # fine-tune GPT-2 (about 3 minutes on a laptop CPU)
python -m app.cli evaluate                   # accuracy on the hand-written test set vs a keyword baseline
python -m app.cli classify "Is the rooftop free on 14 Feb?" --save
python -m app.cli interactive --save         # paste requests one by one
python -m app.cli review                     # saved requests as work queues
pytest
```

The trained model (about 500 MB) is not committed; `train` rebuilds it into `models/request-sorter`.

## How it works (`app/core.py`)

**1. Types and data.** `LABELS` defines the 8 types. `make_training_set` fills sentence templates (`TEMPLATES`)
with random venues, events, dates, dishes, themes and equipment. 30% of the requests are lower-cased, because real
messages are not always tidy. The 40 test requests in `dataset/test_requests.csv` were written separately in a
different, more casual style ("hey, is the rooftop terrace free on the 14th of feb..."), so the test measures
generalisation rather than template matching. A test checks that no test request appears in the training set.

**2. GPT-2 with custom tokens.** `load_base` adds `<|venue_inquiry|>` ... `<|cancellation|>` and `<|pad|>` to
the tokenizer and resizes the embedding matrix. `encode_example` builds `Request: <text>\nType: <|label|><eos>`
and sets the training label to `-100` everywhere except the label token and end-of-text, so the model is only
taught *which type to write*, not how to reproduce the request.

**3. Training.** `train` runs the Hugging Face `Trainer` (warm-up, weight decay, a padding collator that also
masks padded label positions) and saves the model, tokenizer and loss log.

**4. Prediction.** `classify` runs the prompt once and reads the logits of the next token, but keeps only the 8
label tokens and applies a softmax over them. The answer is therefore always a valid type, and you also get a
confidence and a runner-up.

**5. Saving and review.** `save_classification` appends `{time, request, type, confidence, runner_up}` as one
JSON line. `review` groups saved requests by type; anything below 60% confidence goes to `needs_review` for a
person to check.

## Results

| model | accuracy on 40 hand-written requests |
|---|---:|
| keyword baseline (`keyword_baseline`) | 92.5% |
| **fine-tuned GPT-2** | **97.5%** |

Recall is 100% for 7 of the 8 types. The one mistake: *"The board meeting has to be brought forward to Monday
morning."* was predicted `venue_inquiry` instead of `date_change`. "Brought forward" never appears in the
templates, which only say move, reschedule, postpone and shift.

The keyword baseline looks strong because I wrote both its keyword lists and the test set, so it is an optimistic
baseline. GPT-2 still beats it, and it handles requests with no keyword at all (e.g. *"Three of the invitees
dropped out, so it's 47 now."*).

### What went wrong first, and the fix

The first training run predicted the **same type for every request** (12.5% accuracy, 1 in 8). The training loss
was stuck at 1.04, which is exactly `ln(8) / 2`: the loss averages two tokens, end-of-text (learned perfectly) and
the label (still uniform over 8 types). The cause was the new label tokens. After `resize_token_embeddings`, all 8
start at the *same* vector (the average embedding), and AdamW moves each weight by only about the learning rate
per step. After 90 steps the 8 label embeddings were still within 0.03 of each other, so the model could not tell
them apart. Raising the learning rate just made it collapse onto a different single type.

The fix is `init_label_embeddings`: each label token starts at the average embedding of its own words
(`<|venue_inquiry|>` = mean of `" venue"` and `" inquiry"`). The labels then start distinct and meaningful, the
loss drops to 0.001 within 6 epochs, and test accuracy goes from 12.5% to 97.5%. A test now checks this
initialisation.

## Limitations

- 240 templated training requests are a small, narrow sample. Real inbox data (anonymised) would matter most.
- One type per request: "Move the dinner to Sunday and add 10 guests" is really two requests.
- 40 test requests means each mistake moves accuracy by 2.5 points.

Built for the askThili drill *Fine-Tune GPT-2 to Tag Legal Contract Clauses*. Written with help from an AI coding
assistant.
