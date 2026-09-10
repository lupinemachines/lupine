#!/usr/bin/env python3
"""Validate topology assignments and sparse platform/workload selections."""

import argparse
import json
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parent
MODES = ("enumeration", "memory", "streams", "kernel", "peer-access", "peer-copy", "peer-copy-async")


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


def load_specs(topologies=ROOT / "topologies.yaml", runs=ROOT / "runs.yaml"):
    matrix = yaml.safe_load(topologies.read_text())
    selections = yaml.safe_load(runs.read_text())
    if matrix["version"] != 1 or selections["version"] != 1:
        raise ValueError("unsupported spec version")
    for name, topology in matrix["topologies"].items():
        check_topology(matrix["hosts"], topology)
    for name, run in selections["runs"].items():
        if set(run["hosts"]) != set(matrix["hosts"]):
            raise ValueError(f"{name}: platform assignments must cover every host")
        for role, platform in run["hosts"].items():
            platform = selections["platforms"][platform]
            if platform["os"] != "ubuntu" or platform["version"] not in ("22.04", "24.04"):
                raise ValueError(f"{name}/{role}: OS has no provisioning/runner support yet")
            if platform["arch"] not in ("x86_64", "arm64"):
                raise ValueError(f"{name}/{role}: unsupported architecture")
            if matrix["hosts"][role]["gpus"]:
                if platform["arch"] != "x86_64" or "gpu_image" not in platform:
                    raise ValueError(f"{name}/{role}: GCP L4 servers require an x86_64 GPU image")
            elif "cpu_image" not in platform or "cpu_machine" not in platform:
                raise ValueError(f"{name}/{role}: CPU machine and image are required")
        if not run["zones"] or not run["cases"]:
            raise ValueError(f"{name}: zones and cases must not be empty")
        for topology, modes in run["cases"].items():
            if topology not in matrix["topologies"]:
                raise ValueError(f"{name}: unknown topology {topology}")
            if not modes or len(modes) != len(set(modes)) or any(mode not in MODES for mode in modes):
                raise ValueError(f"{name}/{topology}: select nonempty, unique workload modes from {MODES}")
    return matrix, selections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--github-matrix", action="store_true")
    args = parser.parse_args()
    try:
        matrix, selections = load_specs()
    except (KeyError, TypeError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    if args.github_matrix:
        arches = sorted({selections["platforms"][platform]["arch"]
                         for run in selections["runs"].values() for platform in run["hosts"].values()})
        builds = [{"arch": arch, "runner": "ubuntu-24.04-arm" if arch == "arm64" else "ubuntu-24.04"}
                  for arch in arches]
        print("builds=" + json.dumps({"include": builds}))
        print("runs=" + json.dumps({"run": list(selections["runs"])}))
    else:
        print(f"Validated {len(matrix['topologies'])} L4 topologies and {len(selections['runs'])} platform runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
