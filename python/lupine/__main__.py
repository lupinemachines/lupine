"""Command-line entry point for the Lupine Python package."""

from __future__ import annotations

import argparse
import os
import sys

from . import LupineError, login
from ._credentials import DEFAULT_PROFILE, delete_token, profile_name
from ._login import DEFAULT_API_URL, DEFAULT_CONSOLE_URL


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lupine")
    subcommands = parser.add_subparsers(dest="command", required=True)
    profile = argparse.ArgumentParser(add_help=False)
    profile.add_argument(
        "--profile",
        help="credential profile (default: 'default')",
    )

    login_parser = subcommands.add_parser(
        "login",
        help="authenticate with Lupine Cloud in a browser",
        parents=[profile],
    )
    login_parser.add_argument(
        "--api-url", default=os.environ.get("LUPINE_API_URL", DEFAULT_API_URL)
    )
    login_parser.add_argument(
        "--console-url",
        default=os.environ.get("LUPINE_CONSOLE_URL", DEFAULT_CONSOLE_URL),
    )

    logout_parser = subcommands.add_parser(
        "logout",
        help="remove the stored Lupine Cloud credential",
        parents=[profile],
    )
    logout_parser.add_argument(
        "--api-url", default=os.environ.get("LUPINE_API_URL", DEFAULT_API_URL)
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        name = profile_name(args.profile)
        if args.command == "login":
            result = login(
                api_url=args.api_url, console_url=args.console_url, profile=args.profile
            )
            identity = result.email or result.name
            suffix = f" (profile {name})" if name != DEFAULT_PROFILE else ""
            print(
                f"Logged in to Lupine Cloud{f' as {identity}' if identity else ''}"
                f"{suffix}."
            )
        elif args.command == "logout":
            delete_token(args.api_url, args.profile)
            print("Logged out of Lupine Cloud.")
    except (LupineError, OSError) as exc:
        print(f"lupine: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
