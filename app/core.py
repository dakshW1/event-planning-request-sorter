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
# Hint: Extend the GPT-2 tokenizer and configure the model for custom tokens, as shown in the drill for special tags.
def milestone_2():  # TODO(milestone 2)
    raise NotImplementedError("TODO(milestone 2)")


# --- Milestone 3: Train the model on the example requests using the Hugging Face Trainer.
# Hint: Use the Hugging Face Trainer to fine-tune your GPT-2 model on your dataset of event requests.
def milestone_3():  # TODO(milestone 3)
    raise NotImplementedError("TODO(milestone 3)")


# --- Milestone 4: Build an interface (CLI) to input new requests and show their classification.
# Hint: Implement the inference pipeline to take new requests and predict their classification, mirroring the prediction step in the drill.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: Allow saving classified requests to a file for review.
# Hint: Add functionality to your CLI to save the input request and its predicted classification to a file.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")
