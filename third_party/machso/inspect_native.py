#!/usr/bin/env python3
"""Export a native dylib interface for portable ELF dependency discovery."""
import argparse
import json
from aot.macho import native_interface
from aot.native_abi import validate_signatures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('library')
    parser.add_argument('output')
    parser.add_argument('--soname', required=True)
    parser.add_argument('--signatures', help='JSON mapping of fixed scalar function signatures')
    args = parser.parse_args()
    interface = native_interface(args.library, args.soname)
    if args.signatures:
        with open(args.signatures) as source:
            interface['signatures'] = json.load(source)
        validate_signatures(interface)
    with open(args.output, 'w') as output:
        json.dump(interface, output, indent=2)
        output.write('\n')


if __name__ == '__main__':
    main()
