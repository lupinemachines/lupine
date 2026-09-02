"""Command-line entry point for the Lupine Python package."""

from __future__ import annotations

import argparse
import os
import sys

from . import LupineError, login
from ._credentials import delete_token
from ._login import DEFAULT_API_URL, DEFAULT_CONSOLE_URL


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lupine")
    subcommands = parser.add_subparsers(dest="command", required=True)

    login_parser = subcommands.add_parser(
        "login", help="authenticate with Lupine Cloud in a browser"
    )
    login_parser.add_argument(
        "--api-url", default=os.environ.get("LUPINE_API_URL", DEFAULT_API_URL)
    )
    login_parser.add_argument(
        "--console-url",
        default=os.environ.get("LUPINE_CONSOLE_URL", DEFAULT_CONSOLE_URL),
    )

    logout_parser = subcommands.add_parser(
        "logout", help="remove the stored Lupine Cloud credential"
    )
    logout_parser.add_argument(
        "--api-url", default=os.environ.get("LUPINE_API_URL", DEFAULT_API_URL)
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "login":
            result = login(api_url=args.api_url, console_url=args.console_url)
            identity = result.email or result.name
            print(f"Logged in to Lupine Cloud{f' as {identity}' if identity else ''}.")
        elif args.command == "logout":
            delete_token(args.api_url)
            print("Logged out of Lupine Cloud.")
    except (LupineError, OSError) as exc:
        print(f"lupine: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
