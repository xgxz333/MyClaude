from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
from typing import Sequence


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
    args = parser.parse_args(argv)

    if args.command == "ping":
        from my_claude.cli.commands.ping import main as ping_main

        return ping_main(args)
    else:
        parser.print_help()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
