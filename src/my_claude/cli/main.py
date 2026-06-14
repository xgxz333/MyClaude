"""Top-level CLI parser and subcommand dispatcher for `myclaude`."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version


def main(argv: Sequence[str] | None = None) -> int:
    try:
        package_version = version("MyClaude")
    except PackageNotFoundError:
        package_version = "0.1.0"

    parser = argparse.ArgumentParser(prog="myclaude")
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {package_version}",
    )

    subparsers = parser.add_subparsers(dest="command")
    ping_parser = subparsers.add_parser("ping", help="Run a ping check.")
    ping_parser.add_argument("--host", help="Core TCP host.")
    ping_parser.add_argument("--port", type=int, help="Core TCP port.")
    ping_parser.add_argument("--timeout", type=float, help="Ping timeout in seconds.")
    run_parser = subparsers.add_parser("run", help="Run a goal.")
    run_parser.add_argument("--goal", required=True, help="Goal to run.")
    chat_parser = subparsers.add_parser(
        "chat",
        help="Open an interactive chat session.",
        description="Open an interactive chat session.",
    )
    chat_parser.add_argument("--host", help="Core TCP host.")
    chat_parser.add_argument("--port", type=int, help="Core TCP port.")
    chat_parser.add_argument("--timeout", type=float, help="IPC timeout in seconds.")
    chat_parser.add_argument("--session-id", help="Attach to an existing daemon chat session.")
    chat_parser.add_argument("--title", help="Title for a new chat session.")
    trace_parser = subparsers.add_parser("trace", help="View system trace log.")
    trace_parser.add_argument("run_id", nargs="?", default=None, help="Filter by run ID.")
    trace_parser.add_argument("--run-id", dest="run_id_flag", help="Filter by run ID.")
    trace_parser.add_argument("--path", help="Read trace records from this JSONL file.")
    trace_parser.add_argument("--layer", choices=["ipc", "event", "llm"], help="Filter by layer.")
    trace_parser.add_argument("--direction", help="Filter by direction, e.g. CORE→LLM.")
    trace_parser.add_argument("--raw", action="store_true", help="Output raw NDJSON.")
    trace_parser.add_argument("--follow", "-f", action="store_true", help="Follow new records.")
    trace_parser.add_argument("--limit", type=int, help="Maximum number of records to show.")
    trace_parser.add_argument("--json", action="store_true", help="Print raw JSONL records.")
    args = parser.parse_args(argv)

    if args.command == "ping":
        from my_claude.cli.commands.ping import main as ping_main

        return ping_main(args)
    elif args.command == "run":
        from my_claude.cli.commands.run import main as run_main

        return run_main(args)
    elif args.command == "chat":
        from my_claude.cli.commands.chat import main as chat_main

        return chat_main(args)
    elif args.command == "trace":
        from my_claude.cli.commands.trace import main as trace_main

        return trace_main(args)
    else:
        parser.print_help()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
