"""Tests: one block per milestone in README.md.

Training tests use `sshleifer/tiny-gpt2` (a 2-layer GPT-2 with the real GPT-2 tokenizer, ~2 MB) so CI
runs in seconds; the real model is plain `gpt2`.
"""
import json

import pytest

from app import core

TINY = "sshleifer/tiny-gpt2"


# Milestone 1: Define common event request types and create a small dataset of example requests.
def test_training_set_is_balanced_unique_and_reproducible():
    rows = core.make_training_set(per_label=10)
    assert len(rows) == 10 * len(core.LABELS)
    assert {r["label"] for r in rows} == set(core.LABELS)
    assert len({r["text"] for r in rows}) == len(rows)
    assert rows == core.make_training_set(per_label=10)
    assert all("{" not in r["text"] for r in rows)  # every slot was filled


def test_dataset_files_load_and_test_set_is_separate():
    train, test = core.load_csv(core.TRAIN_CSV), core.load_csv(core.TEST_CSV)
    assert len(train) >= 200 and len(test) >= 40
    assert {r["label"] for r in test} == set(core.LABELS)
    assert not {r["text"].lower() for r in train} & {r["text"].lower() for r in test}


def test_csv_round_trip_rejects_unknown_labels(tmp_path):
    core.save_csv([{"text": "hi", "label": "venue_inquiry"}], tmp_path / "a.csv")
    assert core.load_csv(tmp_path / "a.csv") == [{"text": "hi", "label": "venue_inquiry"}]
    core.save_csv([{"text": "hi", "label": "spaceship"}], tmp_path / "b.csv")
    with pytest.raises(ValueError):
        core.load_csv(tmp_path / "b.csv")


# Milestone 2: Configure GPT-2 with an extended tokenizer and model for custom tokens.
@pytest.mark.skip(reason="remove this line when you start milestone 2")
def test_milestone_2():
    raise NotImplementedError("write the test for milestone 2")


# Milestone 3: Train the model on the example requests using the Hugging Face Trainer.
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Build an interface (CLI) to input new requests and show their classification.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


# Milestone 5: Allow saving classified requests to a file for review.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")

