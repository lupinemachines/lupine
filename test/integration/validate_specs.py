#!/usr/bin/env python3
"""Check GPU assignments and expected enumeration for the L4 topology cases."""

from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parent


def check_topology(hosts, topology):
    def devices(host, slots):
        if host not in hosts:
            raise ValueError(f"unknown host {host}")
        for slot in slots:
            if type(slot) is not int or not 0 <= slot < hosts[host]["gpus"]:
                raise ValueError(f"GPU slot {host}/{slot} exceeds the host's GPU count")
        return [f"{host}/{slot}" for slot in slots]

    client = topology["client"]
    servers = topology["servers"]
    if set(client["servers"]) != set(servers) or len(client["servers"]) != len(servers):
        raise ValueError("client.servers must reference every server exactly once")

    expected = devices(client["host"], client["local_gpus"])
    for name in client["servers"]:
        server = servers[name]
        expected.extend(devices(server["host"], server["gpus"]))
    if len(expected) != len(set(expected)):
        raise ValueError("a physical GPU is assigned to multiple processes")
    if topology["expect_devices"] != expected:
        raise ValueError(f"expect_devices must match local then remote order: {expected}")


def main():
    matrix = yaml.safe_load((ROOT / "topologies.yaml").read_text())
    failures = []
    for name, topology in matrix["topologies"].items():
        try:
            check_topology(matrix["hosts"], topology)
        except (KeyError, TypeError, ValueError) as error:
            failures.append(f"{name}: {error}")

    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"Validated {len(matrix['topologies'])} L4 topologies")
    return 0


if __name__ == "__main__":
    sys.exit(main())
