#!/usr/bin/env python3
"""Sign a linked dylib and update its report with the signed artifact's hash."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sign_bundle(output, identity='-'):
    output = Path(output).resolve()
    report_path = Path(str(output)+'.json')
    report = json.loads(report_path.read_text())
    if report['build_status'] not in ('linked', 'signed'):
        raise ValueError('only a successfully linked artifact can be signed')
    subprocess.run(['codesign', '--force', '--sign', identity, str(output)], check=True)
    subprocess.run(['codesign', '--verify', '--strict', str(output)], check=True)
    report['build_status'] = 'signed'
    report['signed'] = True
    report['output_sha256'] = hashlib.sha256(output.read_bytes()).hexdigest()
    report_path.write_text(json.dumps(report, indent=2)+'\n')
    print(f'Signed and verified {output}')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--identity', default='-', help='codesign identity; default is ad-hoc signing')
    args = parser.parse_args()
    try:
        sign_bundle(args.output, args.identity)
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'sign_bundle: {error}\n')


if __name__ == '__main__':
    main()
