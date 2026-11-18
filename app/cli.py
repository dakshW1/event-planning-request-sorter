"""Command-line entry point: `python -m app.cli --help`."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app import core


def _show(text: str, ranked: list[tuple[str, float]], top: int = 3) -> None:
    label, p = ranked[0]
    print(f"\n  {text}\n  -> {label}  ({p:.0%})   {core.LABELS[label]}")
    for lb, q in ranked[1:top]:
        print(f"     also: {lb} ({q:.0%})")


def interactive(model, tok, save_to: Path | None, inp=input, out=print) -> list[dict]:
    """Paste requests one per line; each is classified (and saved if save_to is set). Empty line quits."""
    out("Paste a client request and press Enter. Empty line to quit.")
    saved = []
    while True:
        try:
            text = inp("request> ").strip()
        except EOFError:
            text = ""
        if not text:
            return saved
        ranked = core.classify(model, tok, text)
        _show(text, ranked)
        if save_to:
            saved.append(core.save_classification(text, ranked, save_to))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="event-planning-request-sorter", description="Event planners use this tool to automatically sort incoming client requests (e.g., \"venue inquiry\", \"catering request\") for efficient task delegation and response prioritization.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("types", help="list the request types")
    p = sub.add_parser("dataset", help="(re)build the templated training set")
    p.add_argument("--per-type", type=int, default=30)

    p = sub.add_parser("train", help="fine-tune GPT-2 on the training set with the Hugging Face Trainer")
    p.add_argument("--base", default="gpt2", help="base model on the Hugging Face hub")
    p.add_argument("--epochs", type=float, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--model-dir", type=Path, default=core.MODEL_DIR)

    p = sub.add_parser("classify", help="classify one or more requests")
    p.add_argument("requests", nargs="+", help="request text(s), quoted")
    p.add_argument("--save", action="store_true", help=f"append results to {core.LOG_FILE.name}")
    p.add_argument("--log", type=Path, default=core.LOG_FILE)
    p.add_argument("--model-dir", type=Path, default=core.MODEL_DIR)

    p = sub.add_parser("interactive", help="classify requests as you paste them")
    p.add_argument("--save", action="store_true")
    p.add_argument("--log", type=Path, default=core.LOG_FILE)
    p.add_argument("--model-dir", type=Path, default=core.MODEL_DIR)

    p = sub.add_parser("evaluate", help="accuracy on the hand-written test set vs a keyword baseline")
    p.add_argument("--model-dir", type=Path, default=core.MODEL_DIR)

    p = sub.add_parser("review", help="show saved requests grouped into work queues")
    p.add_argument("--log", type=Path, default=core.LOG_FILE)
    p.add_argument("--min-confidence", type=float, default=0.6)

    args = parser.parse_args(argv)

    if args.command == "types":
        for lb, desc in core.LABELS.items():
            print(f"{lb:<18} {desc}")
    elif args.command == "dataset":
        rows = core.make_training_set(args.per_type)
        core.save_csv(rows, core.TRAIN_CSV)
        print(f"{len(rows)} training requests ({args.per_type} per type) -> {core.TRAIN_CSV.relative_to(core.ROOT)}")
    elif args.command == "train":
        rows = core.load_csv(core.TRAIN_CSV)
        _, _, history = core.train(rows, args.base, args.model_dir, args.epochs, args.lr)
        for h in history:
            print(f"epoch {h['epoch']:.2f}  loss {h['loss']:.4f}")
        print(f"saved to {args.model_dir}")
    elif args.command in ("classify", "interactive"):
        model, tok = core.load_classifier(args.model_dir)
        if args.command == "interactive":
            interactive(model, tok, args.log if args.save else None)
            return
        for text in args.requests:
            ranked = core.classify(model, tok, text)
            _show(text, ranked)
            if args.save:
                core.save_classification(text, ranked, args.log)
        if args.save:
            print(f"\nsaved to {args.log}")
    elif args.command == "evaluate":
        model, tok = core.load_classifier(args.model_dir)
        rows = core.load_csv(core.TEST_CSV)
        result = core.evaluate(model, tok, rows)
        base = sum(core.keyword_baseline(r["text"]) == r["label"] for r in rows) / len(rows)
        print(f"fine-tuned GPT-2 accuracy: {result['accuracy']:.1%}  ({result['n']} hand-written requests)")
        print(f"keyword baseline accuracy: {base:.1%}")
        print("recall per type:", json.dumps(result["recall"]))
        for m in result["mistakes"]:
            print(f"  wrong: {m['text']!r}  true={m['label']}  predicted={m['predicted']}")
    elif args.command == "review":
        queues = core.review(core.load_saved(args.log), args.min_confidence)
        if not queues:
            print(f"nothing saved yet in {args.log}")
        for queue, items in sorted(queues.items()):
            print(f"\n{queue} ({len(items)})")
            for r in items:
                print(f"  [{r['confidence']:.0%}] {r['request']}")


if __name__ == "__main__":
    main()
