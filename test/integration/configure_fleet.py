#!/usr/bin/env python3
"""Resolve a selected platform run to pinned GCP images and Terraform inputs."""

import argparse
import ipaddress
import json
from pathlib import Path
import subprocess

from validate_specs import load_specs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--public-key", type=Path, required=True)
    parser.add_argument("--runner-ip", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    matrix, selections = load_specs()
    run = selections["runs"][args.run]
    hosts = {}
    images = {}
    for role, name in run["hosts"].items():
        platform = selections["platforms"][name]
        gpu = matrix["hosts"][role]["gpus"] != 0
        image = platform["gpu_image" if gpu else "cpu_image"]
        key = (image["project"], image["family"])
        if key not in images:
            result = subprocess.run(
                ["gcloud", "compute", "images", "describe-from-family", image["family"],
                 f"--project={image['project']}", "--format=json"],
                text=True, stdout=subprocess.PIPE, check=True)
            images[key] = json.loads(result.stdout)
        resolved = images[key]
        expected_arch = "ARM64" if platform["arch"] == "arm64" else "X86_64"
        # Older x86 images omit architecture; Arm images declare ARM64.
        if resolved.get("architecture", "X86_64") != expected_arch:
            raise ValueError(f"{role}: image architecture does not match {platform['arch']}")
        hosts[role] = {
            "os": platform["os"], "version": platform["version"], "arch": platform["arch"],
            "machine_type": "g2-standard-24" if gpu else platform["cpu_machine"],
            "image": f"projects/{image['project']}/global/images/{resolved['name']}",
        }
    config = {
        "project_id": args.project, "run_id": args.run_id,
        "host_platforms": hosts,
        "ssh_public_key": args.public_key.read_text(),
        "ssh_source_ranges": [str(ipaddress.IPv4Address(args.runner_ip)) + "/32"],
        "max_run_seconds": 7200,
    }
    args.output.write_text(json.dumps(config, indent=2) + "\n")
    print(" ".join(run["zones"]))


if __name__ == "__main__":
    main()
