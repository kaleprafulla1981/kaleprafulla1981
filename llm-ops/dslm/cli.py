"""Command line: validate, show the chat template, train, query, and serve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dslm.config import load_config
from dslm.data import DataError
from dslm.template import sample_training_text


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default="configs/smollm2-135m-instruct.yaml",
        help="Path to a model and training YAML file.",
    )


def _print_sample(model_name: str, tokenizer=None) -> dict:
    sample = sample_training_text(model_name, tokenizer)
    print(f"Selected model: {sample['model_name']}")
    print(f"Template source: {sample['source']}")
    if sample.get("notes"):
        print(sample["notes"])
    print()
    print("Training row (messages JSON):")
    print(json.dumps({"messages": sample["messages"]}, indent=2, ensure_ascii=False))
    print()
    print("Rendered with the model chat template:")
    print(sample["rendered"].rstrip("\n"))
    print()
    print("Generation continues from this prefix:")
    print(sample["generation_prefix"].rstrip("\n"))
    return sample


def cmd_template(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    tokenizer = None
    if args.live:
        from dslm.train import load_tokenizer

        tokenizer = load_tokenizer(cfg["model"]["name"], cfg["model"].get("trust_remote_code", False))
    sample = _print_sample(cfg["model"]["name"], tokenizer)
    if args.write:
        destination = Path(args.write)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(sample["rendered"], encoding="utf-8")
        print(f"\nWrote {destination}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    from dslm.train import validate

    cfg = load_config(args.config)
    validate(cfg)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from dslm.train import train

    cfg = load_config(args.config)
    if args.output_dir:
        cfg["train"]["output_dir"] = args.output_dir
    if args.max_steps is not None:
        cfg["train"]["max_steps"] = args.max_steps
    train(cfg)
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    from dslm.query import load_bundle, resolve_run

    cfg = load_config(args.config)
    run = resolve_run(cfg, args.run)
    bundle = load_bundle(cfg, run)
    messages = bundle.prepare_messages(question=args.question, system=args.system)
    if args.show_prompt:
        print(bundle.render(messages, add_generation_prompt=True))
        print("---")
    result = bundle.generate(messages)
    print(result["answer"])
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from dslm.query import resolve_run
    from dslm.server import create_app

    cfg = load_config(args.config)
    run = resolve_run(cfg, args.run)
    app = create_app(cfg, str(run) if run else None)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dslm",
        description="Train a domain model, save the adapter, and query it with the model's chat template.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    template = sub.add_parser("template", help="Print the selected model's chat template sample.")
    _add_config(template)
    template.add_argument("--live", action="store_true", help="Load the tokenizer and use its template.")
    template.add_argument("--write", help="Also write the rendered sample to this path.")
    template.set_defaults(func=cmd_template)

    validate = sub.add_parser("validate", help="Check JSONL rows and print one rendered example.")
    _add_config(validate)
    validate.set_defaults(func=cmd_validate)

    train = sub.add_parser("train", help="LoRA-finetune and save the adapter plus a manifest.")
    _add_config(train)
    train.add_argument("--output-dir", help="Override train.output_dir.")
    train.add_argument("--max-steps", type=int, help="Override train.max_steps.")
    train.set_defaults(func=cmd_train)

    query = sub.add_parser("query", help="Ask a question of a saved run, or of the base model.")
    _add_config(query)
    query.add_argument("--run", help="Run directory that contains manifest.json.")
    query.add_argument("--question", required=True)
    query.add_argument("--system", help="Override the system prompt.")
    query.add_argument("--show-prompt", action="store_true")
    query.set_defaults(func=cmd_query)

    serve = sub.add_parser("serve", help="HTTP API used by the query studio.")
    _add_config(serve)
    serve.add_argument("--run", help="Run directory that contains manifest.json.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=43123)
    serve.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (DataError, FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")
