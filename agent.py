#!/usr/bin/env python3
"""CLI entry point: python agent.py --repo <github_repository_url> --output report.json

Runs unattended: clones the repository into an isolated temp directory, lets an LLM drive the
inspection tools, validates every finding, writes the JSON report and cleans up. Exit codes:
0 success, 1 unexpected error, 2 invalid input/configuration, 3 clone failure, 4 LLM failure.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable, Optional, Sequence

from auditor.llm import LLMClient, OpenAIClient
from auditor.loop import AuditAgent, TraceWriter
from auditor.providers import CHOICES, resolve_provider
from auditor.repo import (
    EXIT_BAD_INPUT,
    AuditError,
    RepoRef,
    clone_repo,
    make_workdir,
    parse_repo_url,
    remove_tree,
)
from auditor.report import write_report
from tools import RepoTools


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="agent.py",
        description="AI agent that audits a public GitHub repository for DevOps best-practice violations.",
    )
    p.add_argument("--repo", required=True, help="Public GitHub repository URL, e.g. https://github.com/owner/name")
    p.add_argument("--output", default="report.json", help="Path of the JSON report to write (default: report.json)")
    p.add_argument("--provider", choices=CHOICES, default=None,
                   help="LLM provider (default: $LLM_PROVIDER, else auto = the provider whose API key is set)")
    p.add_argument("--model", default=None, help="LLM model (default: $OPENAI_MODEL / $GEMINI_MODEL, else the provider's default)")
    p.add_argument("--max-steps", type=int, default=40, help="Maximum LLM turns (default: 40)")
    p.add_argument("--timeout", type=int, default=900, help="Wall-clock budget in seconds for the investigation (default: 900)")
    p.add_argument("--clone-depth", type=int, default=200, help="Git history depth to fetch (default: 200)")
    p.add_argument("--trace", default=None, help="Tool-call trace (JSONL) path (default: <output>.trace.jsonl)")
    p.add_argument("--no-trace", action="store_true", help="Do not write a trace file")
    p.add_argument("--quiet", action="store_true", help="Suppress progress output on stderr")
    p.add_argument("--allow-local-repo", action="store_true", help=argparse.SUPPRESS)  # development/testing only
    return p


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # python-dotenv is optional at runtime
        return
    for candidate in (Path.cwd() / ".env", Path(__file__).resolve().parent / ".env"):
        if candidate.is_file():
            load_dotenv(candidate, override=False)


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    llm_factory: Optional[Callable[[str], LLMClient]] = None,
    clone_fn: Callable[..., Path] = clone_repo,
) -> int:
    args = build_parser().parse_args(argv)
    _load_env()
    workdir: Optional[Path] = None
    trace: Optional[TraceWriter] = None
    try:
        if args.max_steps < 3 or args.timeout < 10 or args.clone_depth < 1:
            raise AuditError("--max-steps must be >= 3, --timeout >= 10 and --clone-depth >= 1", EXIT_BAD_INPUT)
        ref: RepoRef = parse_repo_url(args.repo, allow_local=args.allow_local_repo)

        if llm_factory is not None:
            model, provider = args.model or "scripted", args.provider or "test"
            llm = llm_factory(model)
        else:
            cfg = resolve_provider(args.provider, args.model)
            model, provider = cfg.model, cfg.name
            llm = OpenAIClient(api_key=cfg.api_key, model=cfg.model, base_url=cfg.base_url)

        log = (lambda _msg: None) if args.quiet else (lambda msg: print(msg, file=sys.stderr))
        output = Path(args.output)
        trace_path = None if args.no_trace else Path(args.trace) if args.trace else output.with_name(output.stem + ".trace.jsonl")
        trace = TraceWriter(trace_path)

        workdir = make_workdir()
        log(f"[clone] {ref.url} (depth {args.clone_depth})")
        checkout = clone_fn(ref, workdir, depth=args.clone_depth)

        tools = RepoTools(checkout, slug=ref.slug, github_token=os.environ.get("GITHUB_TOKEN") or None)
        log(f"[llm] provider={provider} model={model}")
        trace.event(phase="start", repository=ref.url, provider=provider, model=model, max_steps=args.max_steps)
        agent = AuditAgent(
            llm, tools, ref.url, trace=trace, max_steps=args.max_steps,
            deadline_seconds=args.timeout, log=log,
        )
        violations = agent.run()
        write_report(ref.url, violations, output)
        log(
            f"[done] {len(violations)} violation(s) -> {output} "
            f"({agent.inspection_calls} inspection calls; stopped: {agent.stop_reason or 'finish called'})"
        )
        return 0
    except AuditError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.exit_code
    except KeyboardInterrupt:
        print("error: interrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - last-resort guard: always exit nonzero with a message
        print(f"error: unexpected failure: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if trace is not None:
            trace.close()
        if workdir is not None:
            remove_tree(workdir)


if __name__ == "__main__":
    sys.exit(main())
