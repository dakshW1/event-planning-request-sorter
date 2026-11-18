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
@pytest.fixture(scope="module")
def base():
    return core.load_base(TINY)


def test_each_type_is_one_new_token(base):
    model, tok = base
    ids = core.label_token_ids(tok)
    for label, i in ids.items():
        assert tok.tokenize(core.label_token(label)) == [core.label_token(label)]  # not split up
    assert model.get_input_embeddings().weight.shape[0] == len(tok)  # embeddings resized
    assert tok.pad_token == "<|pad|>"


def test_label_tokens_start_from_their_own_words(base):
    model, tok = base
    emb = model.get_input_embeddings().weight
    ids = core.label_token_ids(tok)
    venue = emb[tok(" venue inquiry")["input_ids"]].mean(dim=0)
    assert (emb[ids["venue_inquiry"]] - venue).abs().max() < 1e-6
    assert (emb[ids["venue_inquiry"]] - emb[ids["cancellation"]]).abs().max() > 0  # distinct from the start


def test_loss_is_only_on_the_label_token(base):
    _, tok = base
    ex = core.encode_example(tok, "Is the hall free on Friday?", "venue_inquiry")
    supervised = [t for t in ex["labels"] if t != -100]
    assert supervised == [core.label_token_ids(tok)["venue_inquiry"], tok.eos_token_id]
    assert ex["input_ids"][-2:] == supervised and len(ex["input_ids"]) == len(ex["labels"])


# Milestone 3: Train the model on the example requests using the Hugging Face Trainer.
@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    out = tmp_path_factory.mktemp("model")
    rows = core.make_training_set(per_label=4)
    model, tok, history = core.train(rows, TINY, out, epochs=3, lr=1e-2, batch_size=8, log_steps=1)
    return model, tok, history, out


def test_trainer_runs_saves_and_reloads(trained):
    model, tok, history, out = trained
    assert history and history[-1]["loss"] < history[0]["loss"]
    assert json.loads((out / "train_log.json").read_text())
    model2, tok2 = core.load_classifier(out)
    assert core.label_token_ids(tok2) == core.label_token_ids(tok)


def test_pad_collator_masks_padding(base):
    _, tok = base
    a = core.encode_example(tok, "short", "cancellation")
    b = core.encode_example(tok, "a much longer request about the projector and the mic", "av_equipment")
    batch = core.PadCollator(tok.pad_token_id)([a, b])
    assert batch["input_ids"].shape == batch["labels"].shape
    pad = batch["attention_mask"][0] == 0
    assert (batch["labels"][0][pad] == -100).all()


# Milestone 4: Build an interface (CLI) to input new requests and show their classification.
def test_classify_returns_a_valid_distribution(trained):
    model, tok, _, _ = trained
    ranked = core.classify(model, tok, "Can you send a quotation for 100 people?")
    assert {lb for lb, _ in ranked} == set(core.LABELS)
    assert abs(sum(p for _, p in ranked) - 1) < 1e-5
    assert ranked == sorted(ranked, key=lambda kv: -kv[1])


def test_keyword_baseline_and_evaluate(trained):
    model, tok, _, _ = trained
    assert core.keyword_baseline("Please cancel the party") == "cancellation"
    rows = core.load_csv(core.TEST_CSV)[:8]
    result = core.evaluate(model, tok, rows)
    assert result["n"] == 8 and 0 <= result["accuracy"] <= 1
    assert sum(sum(c.values()) for c in result["confusion"].values()) == 8


from app.cli import interactive, main  # noqa: E402


def test_cli_types_and_classify(trained, capsys):
    main(["types"])
    assert "venue_inquiry" in capsys.readouterr().out
    main(["classify", "Is there a projector?", "--model-dir", str(trained[3])])
    assert "->" in capsys.readouterr().out


# Milestone 5: Allow saving classified requests to a file for review.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")

