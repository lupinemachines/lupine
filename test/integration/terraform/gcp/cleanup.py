#!/usr/bin/env python3
"""Reconcile a topology run with GCE even when Terraform has forgotten a create."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time


def cleanup(tf_dir, timeout=1200):
    config = json.loads((tf_dir / "run.tfvars.json").read_text())
    project, run_id = config["project_id"], config["run_id"]
    if not project or not re.fullmatch(r"[a-z][a-z0-9-]{0,38}[a-z0-9]", run_id):
        raise ValueError("Cleanup requires a project and a valid, unique run_id")
    if timeout <= 0:
        raise ValueError("Cleanup timeout must be positive")
    deadline = time.monotonic() + timeout
    host_names = {f"{run_id}-{role}" for role in ("a", "b", "client")}
    firewall_names = {f"{run_id}-{role}" for role in ("ssh", "rpc", "deny")}

    def remaining():
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise TimeoutError("Cleanup deadline exceeded; see unresolved resources and operations above")
        return seconds

    def gcloud(*args):
        result = subprocess.run(
            ["gcloud", "compute", *args, f"--project={project}", "--format=json", "--quiet"],
            capture_output=True, text=True, timeout=min(30, remaining()))
        if result.returncode or "Some requests did not succeed" in result.stderr:
            raise RuntimeError(f"gcloud {' '.join(args)} failed: {result.stderr.strip()}")
        return json.loads(result.stdout or "null")

    def target_key(url):
        # Match resource kind, project and complete name, not a shared prefix.
        key = url.partition(f"/projects/{project}/")[2]
        parts = key.split("/")
        if (len(parts) == 4 and parts[0] == "zones" and parts[2] == "instances"
                and parts[3] in host_names):
            return key
        if len(parts) == 3 and parts[:2] == ["global", "firewalls"] and parts[2] in firewall_names:
            return key
        return None

    # A stale lock or a failed create must not prevent the cloud inventory sweep.
    try:
        result = subprocess.run(
            ["terraform", f"-chdir={tf_dir}", "destroy", "-auto-approve",
             "-lock-timeout=30s", "-var-file=run.tfvars.json"],
            timeout=min(300, remaining()))
        if result.returncode:
            print("Terraform destroy failed; reconciling GCE resources directly", flush=True)
    except subprocess.TimeoutExpired:
        print("Terraform destroy timed out; reconciling GCE resources directly", flush=True)

    while True:
        # Read operations before resources so an unfinished create cannot be
        # mistaken for an empty fleet while its instance is not visible yet.
        operations = [operation for operation in gcloud(
            "operations", "list", f"--filter=targetLink~'/{run_id}-'")
            if operation["status"] != "DONE" and target_key(operation.get("targetLink", ""))]
        resources = []
        for kind, names in (("instances", host_names), ("firewall-rules", firewall_names)):
            for resource in gcloud(kind, "list", f"--filter=name~'^{run_id}-'"):
                if resource["name"] not in names:
                    continue
                key = target_key(resource["selfLink"])
                if key is None:
                    raise ValueError(f"Unexpected resource URL: {resource['selfLink']}")
                resources.append((kind, resource, key))

        if not resources and not operations:
            print(f"Cleanup verified: no resources or pending operations for {run_id}", flush=True)
            return

        print("Unresolved cleanup: " + json.dumps({
            "resources": [key for _, _, key in resources],
            "operations": [{field: op.get(field) for field in
                            ("name", "operationType", "status", "targetLink", "error")}
                           for op in operations]}), flush=True)
        deleting = {target_key(op["targetLink"]) for op in operations if op["operationType"] == "delete"}
        for kind, resource, key in resources:
            if key in deleting:
                continue
            args = [kind, "delete", resource["name"]]
            if kind == "instances":
                args.append(f"--zone={key.split('/')[1]}")
            try:
                gcloud(*args)
            except subprocess.TimeoutExpired:
                # gcloud waits for delete completion. Its timeout does not cancel
                # the GCE operation; the next sweep observes it and keeps waiting.
                print(f"Delete still pending for {key}; checking GCE again", flush=True)
            except RuntimeError as error:
                if "was not found" not in str(error):
                    raise
        time.sleep(min(10, remaining()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tf_dir", type=Path)
    parser.add_argument("--timeout", type=float, default=1200, help="Overall cleanup deadline in seconds")
    args = parser.parse_args()
    try:
        cleanup(args.tf_dir, args.timeout)
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Cleanup failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
