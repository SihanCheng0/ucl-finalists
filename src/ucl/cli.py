"""Command line: uv run ucl fetch | build | model | analyze | report | all."""
from __future__ import annotations

import argparse
import sys

from . import config


def cmd_fetch(args: argparse.Namespace) -> int:
    from .dataset import fetch_all
    from .uefa import UefaClient

    failed = fetch_all(UefaClient(), config.SEASONS)
    if failed:
        print(f"{len(failed)} match-stat requests failed; run `uv run ucl fetch` again to resume.", file=sys.stderr)
        return 1
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    from . import dataset
    from .uefa import UefaClient

    ds = dataset.build(UefaClient(), config.SEASONS)
    try:
        dataset.validate(ds)
    except dataset.ValidationError as exc:
        print(f"dataset validation failed:\n{exc}", file=sys.stderr)
        return 2
    dataset.save(ds)
    ts, notes = ds.team_seasons, ds.notes
    print(f"built {len(ts)} team-seasons ({int(ts['in_ko'].sum())} knockout), {len(ds.features)} features")
    if notes["dropped_features"]:
        print("dropped (coverage < 90%):", ", ".join(notes["dropped_features"]))
    print(f"imputed values: {notes['imputed_values']}, coefficients imputed: {notes['coef_imputed']}, "
          f"team-matches without stats: {notes['team_matches_without_stats']}")
    return 0


def cmd_model(args: argparse.Namespace) -> int:
    from . import dataset, model

    ds = dataset.load()
    results = model.run(ds.team_seasons, ds.finals, ds.features)
    model.save(results)
    m = results.metrics
    print(f"Spearman {m['spearman_mean']:.2f} | AUC {m['auc']:.2f} | Brier {m['brier']:.3f} "
          f"(base rate {m['brier_base_rate']:.3f}) | finalists in top 4: {m['finalists_in_top4']:.0%}")
    print(results.drivers.head(config.TOP_DRIVERS)[["feature", "importance", "direction", "label"]]
          .to_string(index=False))
    print(results.ablation.to_string(index=False))
    return 0


COMMANDS = {"fetch": cmd_fetch, "build": cmd_build, "model": cmd_model}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ucl", description="Why the best Champions League teams win.")
    parser.add_argument("command", choices=[*COMMANDS, "all"])
    parser.add_argument("--no-ai", action="store_true", help="skip the local-LLM analysis")
    parser.add_argument("--llm-model", default=config.LLM_MODEL, help="LM Studio model key")
    args = parser.parse_args(argv)
    names = list(COMMANDS) if args.command == "all" else [args.command]
    for name in names:
        code = COMMANDS[name](args)
        if code:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
