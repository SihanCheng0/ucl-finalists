"""Command line: uv run ucl fetch | build | model | analyze | report | all."""
from __future__ import annotations

import argparse
import sys

from . import config


def cmd_fetch(args: argparse.Namespace) -> int:
    from . import stages
    from .uefa import UefaClient

    failed = stages.fetch(UefaClient()).failed
    if failed:
        print(f"{len(failed)} match-stat requests failed; run `uv run ucl fetch` again to resume.", file=sys.stderr)
        return 1
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    from . import dataset, stages
    from .uefa import UefaClient

    try:
        ds = stages.build(UefaClient())
    except dataset.ValidationError as exc:
        print(f"dataset validation failed:\n{exc}", file=sys.stderr)
        return 2
    ts, notes = ds.team_seasons, ds.notes
    print(f"built {len(ts)} team-seasons ({int(ts['in_ko'].sum())} knockout), {len(ds.features)} features")
    if notes["dropped_features"]:
        print("dropped (coverage < 90%):", ", ".join(notes["dropped_features"]))
    print(f"imputed values: {notes['imputed_values']}, coefficients imputed: {notes['coef_imputed']}, "
          f"team-matches without stats: {notes['team_matches_without_stats']}")
    return 0


def _with_ci(m: dict, key: str, fmt: str) -> str:
    lo, hi = m["ci"][key]
    return f"{m[key]:{fmt}} [{lo:{fmt}}, {hi:{fmt}}]"


def cmd_model(args: argparse.Namespace) -> int:
    from . import dataset, stages

    results = stages.model(dataset.load())
    m = results.metrics
    print(f"Spearman {_with_ci(m, 'spearman_mean', '.2f')} | AUC {_with_ci(m, 'auc', '.2f')} | "
          f"Brier skill {_with_ci(m, 'brier_skill', '.2f')} "
          f"(Brier {m['brier']:.3f}, base rate {m['brier_base_rate']:.3f})")
    print(f"finalists in top 4: {_with_ci(m, 'finalists_in_top4', '.0%')}, chance {m['finalists_in_top4_chance']:.0%} "
          "(brackets: 95% intervals from resampling seasons)")
    print(results.drivers.head(config.TOP_DRIVERS)[
        ["feature", "importance", "direction", "marginal_rho", "sign_agree_folds", "label"]
    ].to_string(index=False, float_format="{:.3f}".format))
    print(results.ablation.to_string(index=False, float_format="{:.2f}".format))
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    from . import dataset, model, stages

    analysis = stages.analyze(dataset.load(), model.load(), args.llm_model, not args.no_ai).analysis
    flagged = sum(len(n.unsupported) for n in analysis.narratives.values())
    unavailable = sum(n.status != "ok" for n in analysis.narratives.values())
    print(f"analysis {analysis.status} with {args.llm_model}: {len(analysis.narratives)} narratives "
          f"({unavailable} unavailable), {flagged} figure(s) not found in the data")
    if analysis.status == "unavailable":
        print(f"LM Studio not ready: {analysis.reason or 'unknown reason'}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from . import analyst, dataset, model, stages

    path = stages.report(dataset.load(), model.load(), analyst.load())
    print(f"wrote {path} and report_page.html")
    return 0


COMMANDS = {"fetch": cmd_fetch, "build": cmd_build, "model": cmd_model, "analyze": cmd_analyze,
            "report": cmd_report}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["web"]:  # the dashboard has its own options, so it bypasses the stage parser
        from .web.server import main as web_main

        return web_main(argv[1:])
    parser = argparse.ArgumentParser(
        prog="ucl", description="Why the best Champions League teams win.",
        epilog="ucl web [--port N] [--no-open] starts UCL Lab, the local dashboard.")
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
