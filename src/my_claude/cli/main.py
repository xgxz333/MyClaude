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
    args = parser.parse_args(argv)

    if args.command == "ping":
        from my_claude.cli.commands.ping import main as ping_main

        return ping_main(args)
    elif args.command == "run":
        from my_claude.cli.commands.run import main as run_main

        return run_main(args)
    else:
        parser.print_help()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
