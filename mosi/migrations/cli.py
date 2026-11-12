import argparse
import sys
from typing import Optional, Sequence

from .runner import migrate


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="mosi", description="model-sitter command line tools")
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("migrate", help="convert run folders of an older layout to the current one")
    m.add_argument("path", help="one run folder, or a folder to search for runs")
    m.add_argument("--apply", action="store_true", help="do it (default is a dry run that changes nothing)")
    args = parser.parse_args(argv)
    try:
        counts = migrate(args.path, apply=args.apply)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 1 if counts["refused"] else 0


if __name__ == "__main__":
    sys.exit(main())
