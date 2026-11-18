"""Event Planning Request Sorter: GPT-2 fine-tuned to tag client requests with a request type.

Same recipe as the legal-clause drill: every request type becomes a special token, GPT-2 learns to
write that token after "Request: ... Type:", and at prediction time we compare only the label tokens.

  1. request types + a small dataset (templated training set, hand-written test set)
  2. GPT-2 with an extended tokenizer (one special token per type) and resized embeddings
  3. fine-tuning with the Hugging Face Trainer (loss only on the label token)
  4. inference pipeline used by the CLI (app/cli.py)
  5. saving classified requests to a file for review
"""
from __future__ import annotations

import csv
import json
import math
import random
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
TRAIN_CSV = DATASET_DIR / "train_requests.csv"
TEST_CSV = DATASET_DIR / "test_requests.csv"
MODEL_DIR = ROOT / "models" / "request-sorter"
LOG_FILE = ROOT / "classified_requests.jsonl"


# --- Milestone 1: Define common event request types and create a small dataset of example requests.

LABELS = {
    "venue_inquiry": "availability, capacity or facilities of a venue",
    "catering_request": "food, drinks, menus and dietary needs",
    "budget_quote": "prices, quotations, packages and discounts",
    "date_change": "moving the event to another date or time",
    "guest_list": "headcount changes, attendee lists, RSVPs, badges",
    "decor_theme": "decoration, theme, flowers, colours, lighting look",
    "av_equipment": "projectors, microphones, sound, wifi, streaming",
    "cancellation": "calling off the event, refunds, cancellation policy",
}

# Each type gets a few sentence templates; {slots} are filled from SLOTS below.
TEMPLATES = {
    "venue_inquiry": [
        "Is {venue} available on {date} for {n} guests?",
        "What is the maximum capacity of {venue}?",
        "Can we book {venue} for our {event} on {date}?",
        "Does {venue} have parking and wheelchair access?",
        "We are looking for a venue for a {event} with {n} people. Do you have something?",
        "Could we visit {venue} before confirming the {event}?",
    ],
    "catering_request": [
        "Can you arrange a {food} menu for {n} guests?",
        "We need {meal} for the {event}, some guests are {diet}.",
        "Please send the catering options for a {food} {meal}.",
        "Is it possible to add a {food} counter at the {event}?",
        "Could the caterer prepare {diet} options for the {meal}?",
    ],
    "budget_quote": [
        "How much would a {event} for {n} people cost?",
        "Can you send a detailed quotation for the {event}?",
        "Our budget is {money}. What package can you offer for the {event}?",
        "What is the price difference between {venue} and a smaller hall?",
        "Is there a discount if we book the {event} on a weekday?",
    ],
    "date_change": [
        "Can we move the {event} from {date} to {date2}?",
        "Please reschedule our {event} to {date2}.",
        "We need to change the start time of the {event} to {time}.",
        "Is it possible to postpone the {event} by a week?",
        "Our {event} has to be shifted to {date2}, is {venue} free then?",
    ],
    "guest_list": [
        "Our guest count for the {event} has changed to {n}.",
        "Please find the updated list of attendees for the {event}.",
        "We have {k} more people joining the {event}.",
        "Can you send RSVP reminders to the guests of the {event}?",
        "Please print name cards for all {n} guests.",
    ],
    "decor_theme": [
        "We would like a {theme} theme for the {event}.",
        "Can the decoration use {colour} and {flower}?",
        "Please suggest decor ideas for a {theme} {event}.",
        "We want {flower} on every table and {colour} lighting.",
        "Could the stage backdrop follow a {theme} style?",
    ],
    "av_equipment": [
        "Will there be a {av} available at {venue}?",
        "We need a {av} and a {av2} for the {event}.",
        "Can you set up a {av} for the speakers?",
        "Is the {av} included in the {event} package?",
        "Please arrange a {av} and test it before the {event} starts.",
    ],
    "cancellation": [
        "We have to cancel the {event} on {date}.",
        "Please cancel our booking of {venue}. How much will be refunded?",
        "Unfortunately the {event} is called off.",
        "What is your cancellation policy for the {event}?",
        "We will not need {venue} anymore, please cancel the {event}.",
    ],
}

SLOTS = {
    "venue": ["the rooftop hall", "the garden lawn", "the banquet hall", "the lakeside pavilion",
              "the conference room", "the beach venue"],
    "event": ["wedding reception", "birthday party", "product launch", "conference", "team offsite",
              "engagement ceremony", "charity gala", "workshop"],
    "date": ["12 March", "the first Saturday of May", "next Friday", "25 December", "the 18th"],
    "date2": ["the following Sunday", "20 April", "next month", "the 30th", "a weekday"],
    "time": ["4 pm", "10 am", "7:30 pm", "noon"],
    "n": ["40", "75", "120", "200", "350"],
    "k": ["five", "12", "twenty", "8"],
    "food": ["South Indian", "Italian", "barbecue", "Chinese", "dessert", "mocktail"],
    "meal": ["lunch", "dinner", "breakfast", "high tea", "buffet"],
    "diet": ["vegan", "gluten free", "Jain", "nut allergic", "diabetic"],
    "money": ["two lakh rupees", "5,000 dollars", "around 80,000", "a tight budget"],
    "theme": ["rustic", "royal", "beach", "vintage", "minimal", "Bollywood"],
    "colour": ["gold and white", "pastel pink", "navy blue", "green"],
    "flower": ["roses", "marigolds", "orchids", "lilies"],
    "av": ["projector", "wireless microphone", "sound system", "LED screen", "podium mic", "live stream setup"],
    "av2": ["laptop connection", "spotlight", "speaker", "clicker"],
}


def make_training_set(per_label: int = 30, seed: int = 7) -> list[dict]:
    """Fill every template with random slot values: per_label unique requests for each type."""
    rng = random.Random(seed)
    rows = []
    for label, templates in TEMPLATES.items():
        seen: set[str] = set()
        while len(seen) < per_label:
            tpl = rng.choice(templates)
            text = tpl.format(**{k: rng.choice(v) for k, v in SLOTS.items()})
            if rng.random() < 0.3:
                text = text.lower()  # real requests are not always neatly capitalised
            seen.add(text)
        rows += [{"text": t, "label": label} for t in sorted(seen)]
    rng.shuffle(rows)
    return rows


def save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    unknown = {r["label"] for r in rows} - set(LABELS)
    if unknown:
        raise ValueError(f"unknown labels in {path.name}: {unknown}")
    return rows


# --- Milestone 2: Configure GPT-2 with an extended tokenizer and model for custom tokens.

def label_token(label: str) -> str:
    return f"<|{label}|>"


def prompt_for(text: str) -> str:
    return f"Request: {text.strip()}\nType:"


def load_base(base_model: str = "gpt2"):
    """GPT-2 + tokenizer, extended with one special token per request type and a pad token."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model)
    tok.add_special_tokens({"pad_token": "<|pad|>",
                            "additional_special_tokens": [label_token(lb) for lb in LABELS]})
    model = AutoModelForCausalLM.from_pretrained(base_model)
    model.resize_token_embeddings(len(tok))
    model.config.pad_token_id = tok.pad_token_id
    init_label_embeddings(model, tok)
    return model, tok


def init_label_embeddings(model, tok) -> None:
    """Start each new label token at the average embedding of its own words (<|venue_inquiry|> starts at
    mean(' venue', ' inquiry')). New tokens otherwise all start at the same vector, and since AdamW moves a
    weight by about the learning rate per step, a few hundred steps cannot pull 8 identical vectors apart:
    the model then gives every request the same type. GPT-2 ties input and output embeddings, so this also
    sets the output side."""
    import torch

    emb = model.get_input_embeddings().weight
    with torch.no_grad():
        for label in LABELS:
            word_ids = tok(" " + label.replace("_", " "))["input_ids"]
            emb[tok.convert_tokens_to_ids(label_token(label))] = emb[word_ids].mean(dim=0)


def label_token_ids(tok) -> dict[str, int]:
    ids = {lb: tok.convert_tokens_to_ids(label_token(lb)) for lb in LABELS}
    assert len(set(ids.values())) == len(LABELS) and tok.unk_token_id not in ids.values()
    return ids


def encode_example(tok, text: str, label: str, max_length: int = 64) -> dict:
    """input = prompt + label token + eos; labels = -100 everywhere except the label token and eos,
    so the loss teaches only *which type to write*, not how to reproduce the request."""
    prompt_ids = tok(prompt_for(text))["input_ids"][: max_length - 2]
    target = [tok.convert_tokens_to_ids(label_token(label)), tok.eos_token_id]
    return {"input_ids": prompt_ids + target, "attention_mask": [1] * (len(prompt_ids) + 2),
            "labels": [-100] * len(prompt_ids) + target}


# --- Milestone 3: Train the model on the example requests using the Hugging Face Trainer.

class PadCollator:
    """Right-pad a batch; padded label positions are -100 so they are ignored by the loss."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict:
        import torch

        width = max(len(b["input_ids"]) for b in batch)
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for b in batch:
            pad = width - len(b["input_ids"])
            out["input_ids"].append(b["input_ids"] + [self.pad_id] * pad)
            out["attention_mask"].append(b["attention_mask"] + [0] * pad)
            out["labels"].append(b["labels"] + [-100] * pad)
        return {k: torch.tensor(v) for k, v in out.items()}


def train(rows: list[dict], base_model: str = "gpt2", out_dir: Path = MODEL_DIR, epochs: float = 4,
          lr: float = 5e-5, batch_size: int = 16, seed: int = 42, log_steps: int = 10):
    """Fine-tune with the Hugging Face Trainer, save model + tokenizer, return (model, tok, log_history)."""
    from transformers import Trainer, TrainingArguments, set_seed

    set_seed(seed)
    model, tok = load_base(base_model)
    ds = [encode_example(tok, r["text"], r["label"]) for r in rows]  # a list works as a torch Dataset
    total_steps = math.ceil(len(ds) / batch_size) * math.ceil(epochs)
    args = TrainingArguments(output_dir=str(Path(out_dir) / "_trainer"), num_train_epochs=epochs,
                             learning_rate=lr, per_device_train_batch_size=batch_size,
                             warmup_steps=max(1, total_steps // 10), weight_decay=0.01, logging_steps=log_steps,
                             save_strategy="no", report_to=[], seed=seed, use_cpu=True,
                             disable_tqdm=True)
    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=PadCollator(tok.pad_token_id))
    trainer.train()
    model.eval()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    history = [h for h in trainer.state.log_history if "loss" in h]
    (Path(out_dir) / "train_log.json").write_text(json.dumps(history, indent=1))
    return model, tok, history


def load_classifier(model_dir: Path = MODEL_DIR):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not (Path(model_dir) / "config.json").exists():
        raise FileNotFoundError(f"no trained model in {model_dir}; run `python -m app.cli train` first")
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    model.eval()
    return model, tok



# --- Milestone 4: Build an interface (CLI) to input new requests and show their classification.

def classify(model, tok, text: str) -> list[tuple[str, float]]:
    """All request types ranked by probability. The model scores the token after 'Type:'; we keep
    only the 8 label tokens and softmax over them, so the answer is always a valid type."""
    import torch

    ids = tok(prompt_for(text), return_tensors="pt")
    with torch.no_grad():
        logits = model(**ids).logits[0, -1]
    label_ids = label_token_ids(tok)
    probs = torch.softmax(logits[list(label_ids.values())], dim=-1).tolist()
    return sorted(zip(label_ids, probs), key=lambda kv: -kv[1])


def evaluate(model, tok, rows: list[dict]) -> dict:
    """Accuracy, per-type recall and a confusion count on labelled rows."""
    confusion: dict[str, dict[str, int]] = {lb: {} for lb in LABELS}
    correct = 0
    mistakes = []
    for r in rows:
        pred = classify(model, tok, r["text"])[0][0]
        confusion[r["label"]][pred] = confusion[r["label"]].get(pred, 0) + 1
        if pred == r["label"]:
            correct += 1
        else:
            mistakes.append({"text": r["text"], "label": r["label"], "predicted": pred})
    recall = {lb: round(c.get(lb, 0) / max(sum(c.values()), 1), 2) for lb, c in confusion.items()}
    return {"accuracy": round(correct / len(rows), 3), "n": len(rows), "recall": recall,
            "confusion": confusion, "mistakes": mistakes}


KEYWORDS = {  # a simple keyword baseline, to show what fine-tuning adds
    "venue_inquiry": ["venue", "hall", "capacity", "available", "space", "room", "parking"],
    "catering_request": ["menu", "food", "lunch", "dinner", "caterer", "buffet", "vegan", "snacks"],
    "budget_quote": ["cost", "price", "quote", "quotation", "budget", "discount", "how much"],
    "date_change": ["move", "reschedule", "postpone", "shift", "instead", "later", "forward"],
    "guest_list": ["guest", "attendee", "headcount", "rsvp", "list", "people now", "invitee"],
    "decor_theme": ["theme", "decor", "flower", "colour", "lights", "backdrop", "balloon"],
    "av_equipment": ["projector", "mic", "sound", "wifi", "screen", "dj", "stream", "speaker"],
    "cancellation": ["cancel", "call off", "refund", "withdrawn", "drop the event"],
}


def keyword_baseline(text: str) -> str:
    t = text.lower()
    scores = {lb: sum(w in t for w in words) for lb, words in KEYWORDS.items()}
    return max(scores, key=lambda lb: (scores[lb], lb == "venue_inquiry"))


# --- Milestone 5: Allow saving classified requests to a file for review.
# Hint: Add functionality to your CLI to save the input request and its predicted classification to a file.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")
