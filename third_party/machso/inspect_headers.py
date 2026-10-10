#!/usr/bin/env python3
"""Attach header-derived scalar signatures to an inspected native interface."""
import argparse
import json
import subprocess
from pathlib import Path

from aot.headers import read_signatures
from aot.native_abi import validate_signatures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('interface', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--header', type=Path, action='append', required=True)
    parser.add_argument('-I', '--include-dir', type=Path, action='append', default=[])
    parser.add_argument('--clang', default='clang')
    parser.add_argument('--language', choices=('c', 'c++'), default='c')
    parser.add_argument('--sysroot', type=Path, help='ARM64 Linux header sysroot for cross-host parsing')
    args = parser.parse_args()
    try:
        interface = json.loads(args.interface.read_text())
        report = read_signatures(args.header, interface['exports'], args.include_dir,
                                 args.clang, args.language, args.sysroot)
        interface['signatures'] = report['signatures']
        interface['signature_report'] = {key: value for key, value in report.items() if key != 'signatures'}
        validate_signatures(interface)
        args.output.write_text(json.dumps(interface, indent=2)+'\n')
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'inspect_headers: {error}\n')
    except ImportError as error:
        parser.exit(1, f'inspect_headers: {error}; run with uv --extra headers\n')
    print(f"Read {len(report['signatures'])} scalar signatures; "
          f"{len(report['unsupported'])} unsupported; {len(report['missing_declarations'])} undeclared exports.")


if __name__ == '__main__':
    main()
