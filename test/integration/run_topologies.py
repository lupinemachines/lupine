#!/usr/bin/env python3
"""Run the L4 topology matrix over SSH against Terraform's inventory output."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
import json
from pathlib import Path
import shlex
import subprocess
import time
import xml.etree.ElementTree as ET

from validate_specs import SAMPLES, SAMPLE_SPEC, WORKLOADS, load_specs


class Runner:
    def __init__(self, inventory, key, artifacts, results, tests, jobs=8):
        self.inventory = inventory
        self.hosts = inventory["hosts"]
        self.artifacts = artifacts
        self.results = results
        self.tests = tests
        self.jobs = jobs
        self.remote = f"/tmp/{inventory['run_id']}"
        self.options = ["-i", str(key), "-o", "IdentitiesOnly=yes",
                        "-o", "StrictHostKeyChecking=accept-new",
                        "-o", f"UserKnownHostsFile={results / 'known_hosts'}",
                        "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=10",
                        "-o", "ServerAliveCountMax=2"]
        self.native = {}
        self.cases = []

    def target(self, role):
        host = self.hosts[role]
        return f"{host['ssh_user']}@{host['ssh_host']}"

    def ssh(self, role, command, timeout=70, check=True):
        try:
            result = subprocess.run(
                ["ssh", *self.options, self.target(role), command],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                errors="replace", timeout=timeout)
        except subprocess.TimeoutExpired as error:
            output = error.stdout or b""
            if isinstance(output, bytes):
                output = output.decode(errors="replace")
            result = subprocess.CompletedProcess([], 124, output + "\nSSH timeout\n")
        if check and result.returncode:
            raise RuntimeError(f"{role}: {result.stdout}")
        return result

    def command(self, role, args, env=None, **kwargs):
        environment = ["env", "-u", "LD_PRELOAD", "-u", "LD_LIBRARY_PATH",
                       "-u", "CUDA_VISIBLE_DEVICES", "-u", "LUPINE_SERVER"]
        environment += [f"{name}={value}" for name, value in (env or {}).items()]
        command = f"cd {shlex.quote(self.remote)} && " + shlex.join(environment + args)
        return self.ssh(role, command, **kwargs)

    def deploy(self):
        for role in self.hosts:
            print(f"Waiting for {role} startup", flush=True)
            deadline = time.monotonic() + 600
            while True:
                result = self.ssh(role,
                                  "if test -f /var/lib/lupine-startup-complete; then exit 0; "
                                  "elif test -f /var/lib/lupine-startup-failed; then exit 42; "
                                  "else exit 1; fi",
                                  timeout=15, check=False)
                if result.returncode == 0:
                    break
                if result.returncode == 42:
                    raise RuntimeError(f"{role}: VM startup script failed; inspect its serial log")
                if time.monotonic() > deadline:
                    raise RuntimeError(f"{role}: VM startup did not finish: {result.stdout}")
                time.sleep(5)
            self.ssh(role, shlex.join(["mkdir", "-p", self.remote]))
            # Verify what actually booted, independently of the image name and
            # requested machine type, before selecting the architecture's build.
            probe = self.command(role, ["python3", "-c",
                "import json,platform; release=platform.freedesktop_os_release(); "
                "print(json.dumps(dict(os=release['ID'],version=release['VERSION_ID'],"
                "arch={'aarch64':'arm64'}.get(platform.machine(),platform.machine()))))"])
            actual = json.loads(probe.stdout)
            expected = {key: self.hosts[role][key] for key in ("os", "version", "arch")}
            if actual != expected:
                raise RuntimeError(f"{role}: platform mismatch: got {actual}, expected {expected}")
            self.hosts[role]["observed_platform"] = actual
            artifact = self.artifacts / f"linux-{actual['arch']}.tar.gz"
            subprocess.run(["scp", *self.options, str(artifact),
                            f"{self.target(role)}:{self.remote}/artifacts.tar.gz"],
                           check=True, timeout=120)
            self.command(role, ["tar", "xzf", "artifacts.tar.gz"])
            if role == "client":
                self.ssh(role, "test ! -e /dev/nvidia0")
                continue
            deadline = time.monotonic() + 180
            while True:
                probe = self.command(role, ["./topology_test", "inventory"], check=False)
                if probe.returncode == 0:
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError(f"{role}: native CUDA did not become ready: {probe.stdout}")
                time.sleep(5)
            data = json.loads(probe.stdout)
            expected = self.hosts[role]["expected_gpu_count"]
            if len(data["devices"]) != expected or any(
                    "L4" not in device["name"] for device in data["devices"]):
                raise RuntimeError(f"{role}: expected {expected} L4 GPUs, got {data}")
            # Keep the original native ordinals for the peer matrix; slots use UUID order.
            data["slots"] = sorted(range(expected), key=lambda i: data["devices"][i]["uuid"])
            self.native[role] = data
            topo = self.ssh(role, "nvidia-smi topo -m", check=False)
            (self.results / f"native-{role}-topology.log").write_text(topo.stdout)
        self.save_manifest()

    def gpu(self, reference):
        role, slot = reference.split("/")
        native = self.native[role]
        ordinal = native["slots"][int(slot)]
        return native["devices"][ordinal]["uuid"]

    def native_peer(self, source, destination):
        role, src = source.split("/")
        other, dst = destination.split("/")
        if role != other:
            return 0
        native = self.native[role]
        return native["peer_access"][native["slots"][int(src)]][native["slots"][int(dst)]]

    def save_manifest(self, **extra):
        self.inventory.update(extra)
        data = dict(self.inventory, native=self.native, jobs=self.jobs,
                    samples_revision=SAMPLE_SPEC["revision"])
        (self.results / "resolved.json").write_text(json.dumps(data, indent=2) + "\n")

    def record(self, name, result, elapsed=0, skip_reason=None):
        logfile = name.replace("/", "_") + ".log"
        (self.results / logfile).write_text(result.stdout)
        self.cases.append({"name": name, "returncode": result.returncode,
                           "seconds": elapsed, "log": logfile, "skip_reason": skip_reason})
        status = "SKIP" if skip_reason else "FAIL" if result.returncode else "PASS"
        print(f"{status} {name}", flush=True)
        if result.returncode:
            print(result.stdout[-4000:], flush=True)
        # Persist after each case so an interrupted job retains completed results.
        self.report()

    def test(self, name, role, mode, expected, env, pair=(), peer=None, can_peer=False):
        sample = SAMPLES[mode.removeprefix("sample:")] if mode.startswith("sample:") else None
        args = ["timeout", "--kill-after=5s", "45s", "./topology_test", mode,
                ",".join(self.gpu(device) for device in expected)]
        args.extend(str(value) for value in pair)
        if peer is not None:
            args.append(str(peer))
        executable = f"{self.remote}/topology_test"
        if sample is not None:
            executable = f"{self.remote}/samples/{mode.removeprefix('sample:')}"
            args = ["timeout", "--kill-after=5s", f"{sample.get('timeout', 120)}s",
                    executable, *sample.get("args", [])]
            libraries = [f"{self.remote}/samples", *filter(None, [env.get("LD_LIBRARY_PATH")])]
            env = dict(env, LD_LIBRARY_PATH=":".join(libraries))
        driver = f"{self.remote}/libcuda.so.1" if env.get("LUPINE_SERVER") else self.native[role]["driver_library"]
        guard = f"{self.remote}/driver_guard.so"
        env = dict(env, LUPINE_TEST_EXECUTABLE=executable, LUPINE_TEST_DRIVER=driver,
                   LD_PRELOAD=f"{driver}:{guard}")
        start = time.monotonic()
        preflight = ""
        if sample is not None:
            # The sample's upstream source is unchanged. Verify its exact device
            # map under the same environment before launching it.
            probe = self.command(role, ["timeout", "--kill-after=5s", "45s", "./topology_test",
                                       "enumeration", ",".join(self.gpu(device) for device in expected)],
                                 env, check=False)
            if probe.returncode:
                return probe, time.monotonic() - start, None
            preflight = probe.stdout
        result = self.command(role, args, env, timeout=sample.get("timeout", 120) + 25 if sample else 70,
                              check=False)
        marker = f"LUPINE_TEST_DRIVER_OK {executable} {driver}"
        if marker not in result.stdout.splitlines():
            result = subprocess.CompletedProcess(result.args, result.returncode or 1,
                result.stdout + "\nMissing in-process CUDA driver attestation\n")
            return result, time.monotonic() - start, None
        result.stdout = preflight + result.stdout
        skip_reason = None
        if (result.returncode == 2 and sample and sample.get("peer_waiver")
                and sample["peer_waiver"] in result.stdout and not can_peer):
            skip_reason = "upstream EXIT_WAIVED: no direct peer-access pair in this topology"
        return result, time.monotonic() - start, skip_reason

    def workloads(self, name, role, devices, owners, env, tests):
        cases = []
        can_peer = any(owners[src] == owners[dst] and self.native_peer(devices[src], devices[dst])
                       for src, dst in itertools.permutations(range(len(devices)), 2))
        for mode in tests:
            if not mode.startswith("peer-"):
                cases.append((f"{name}.{mode}", role, mode, devices, env, (), None, can_peer))
        pair_modes = [mode for mode in tests if mode.startswith("peer-")]
        for src, dst in itertools.permutations(range(len(devices)), 2):
            peer = self.native_peer(devices[src], devices[dst]) if owners[src] == owners[dst] else 0
            label = f"{devices[src]}-to-{devices[dst]}"
            for mode in pair_modes:
                cases.append((f"{name}.{mode}.{label}", role, mode, devices, env,
                              (src, dst), peer if mode == "peer-access" else None))
        # Only the collector writes reports. Drain all clients before the caller
        # can stop servers or reassign GPUs for the next topology.
        with ThreadPoolExecutor(max_workers=self.jobs) as pool:
            pending = {pool.submit(self.test, *case): case[0] for case in cases}
            for future in as_completed(pending):
                try:
                    result, elapsed, skip_reason = future.result()
                except Exception as error:
                    result, elapsed, skip_reason = subprocess.CompletedProcess([], 1, str(error)), 0, None
                self.record(pending[future], result, elapsed, skip_reason)

    def baseline(self):
        for role in self.native:
            devices = [f"{role}/{i}" for i in range(len(self.native[role]["slots"]))]
            visible = ",".join(self.gpu(device) for device in devices)
            self.workloads(f"native-{role}", role, devices, ["native"] * len(devices),
                           {"CUDA_VISIBLE_DEVICES": visible}, self.tests)

    def topology(self, name, spec, tests):
        servers = []
        endpoints = {}
        client = spec["client"]
        try:
            for index, (server, config) in enumerate(sorted(spec["servers"].items())):
                role = config["host"]
                port = 14833 + index
                log = f"{name}-{server}.log"
                visible = ",".join(self.gpu(f"{role}/{slot}") for slot in config["gpus"])
                script = (
                    "import os,subprocess,sys; "
                    "env=dict(os.environ,CUDA_VISIBLE_DEVICES=sys.argv[1],LUPINE_PORT=sys.argv[2]); "
                    "p=subprocess.Popen(['./lupine_driver_server'],env=env,"
                    "stdin=subprocess.DEVNULL,stdout=open(sys.argv[3],'w'),"
                    "stderr=subprocess.STDOUT,start_new_session=True); print(p.pid)")
                pid = int(self.command(role, ["python3", "-c", script, visible, str(port), log]).stdout)
                servers.append((role, pid, log))
                endpoints[server] = f"{self.hosts[role]['rpc_host']}:{port}"
            ordered = [endpoints[server] for server in client["servers"]]
            local = [f"{client['host']}/{slot}" for slot in client["local_gpus"]]
            env = {"LUPINE_SERVER": ",".join(ordered),
                   "CUDA_VISIBLE_DEVICES": ",".join(self.gpu(device) for device in local),
                   "LD_LIBRARY_PATH": self.remote}
            owners = ["local"] * len(local)
            for server in client["servers"]:
                owners.extend([server] * len(spec["servers"][server]["gpus"]))
            self.inventory.setdefault("topologies", {})[name] = {
                "spec": spec, "environment": env, "owners": owners,
                "tests": tests,
                "uuids": [self.gpu(device) for device in spec["expect_devices"]]}
            self.save_manifest()
            # Check reachability from the actual client VM, not from the SSH runner.
            script = "import socket,sys; [socket.create_connection((x.rsplit(':',1)[0],int(x.rsplit(':',1)[1])),2).close() for x in sys.argv[1:]]"
            deadline = time.monotonic() + 60
            while self.command(client["host"], ["python3", "-c", script, *ordered], check=False).returncode:
                if time.monotonic() > deadline:
                    raise RuntimeError("servers did not become reachable from the client")
                time.sleep(1)
            self.workloads(name, client["host"], spec["expect_devices"], owners, env, tests)
        except Exception as error:
            self.record(f"{name}.setup", subprocess.CompletedProcess([], 1, str(error)))
        finally:
            for role, pid, log in servers:
                script = (
                    "import os,signal,sys,time; pid=int(sys.argv[1]); "
                    "\ntry:\n os.killpg(pid,signal.SIGTERM); time.sleep(1); os.killpg(pid,signal.SIGKILL)"
                    "\nexcept ProcessLookupError: pass")
                self.command(role, ["python3", "-c", script, str(pid)], check=False)
                result = self.command(role, ["cat", log], check=False)
                (self.results / log).write_text(result.stdout)

    def report(self):
        failures = sum(case["returncode"] != 0 and not case["skip_reason"] for case in self.cases)
        skipped = sum(bool(case["skip_reason"]) for case in self.cases)
        suite = ET.Element("testsuite", name="l4-topologies", tests=str(len(self.cases)),
                           failures=str(failures), skipped=str(skipped),
                           time=str(sum(case["seconds"] for case in self.cases)))
        for case in self.cases:
            node = ET.SubElement(suite, "testcase", name=case["name"], time=str(case["seconds"]))
            output = (self.results / case["log"]).read_text()
            if case["skip_reason"]:
                ET.SubElement(node, "skipped", message=case["skip_reason"])
            elif case["returncode"]:
                ET.SubElement(node, "failure", message=f"exit {case['returncode']}").text = output
            ET.SubElement(node, "system-out").text = output
        ET.ElementTree(suite).write(self.results / "junit.xml", encoding="utf-8", xml_declaration=True)
        (self.results / "results.json").write_text(json.dumps(self.cases, indent=2) + "\n")
        return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--ssh-key", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=Path(__file__).with_name("topologies.yaml"))
    parser.add_argument("--runs-spec", type=Path, default=Path(__file__).with_name("runs.yaml"))
    parser.add_argument("--run", default="regression")
    parser.add_argument("--jobs", type=int, default=8, help="concurrent client processes per topology (default: 8)")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    matrix, selections = load_specs(args.spec, args.runs_spec)
    selection = selections["runs"][args.run]
    inventory = json.loads(args.inventory.read_text())
    if set(inventory["hosts"]) != set(matrix["hosts"]):
        raise ValueError("inventory host roles differ from the topology matrix")
    if len({host["instance_id"] for host in inventory["hosts"].values()}) != len(matrix["hosts"]):
        raise ValueError("machine roles must resolve to distinct instances")
    for role, name in selection["hosts"].items():
        platform = selections["platforms"][name]
        if any(inventory["hosts"][role][key] != platform[key] for key in ("os", "version", "arch")):
            raise ValueError(f"{role}: inventory platform differs from selected run")
        if inventory["hosts"][role]["expected_gpu_count"] != matrix["hosts"][role]["gpus"]:
            raise ValueError(f"{role}: inventory GPU count differs from topology spec")
    inventory["selection"] = {"name": args.run, **selection}
    args.results.mkdir(parents=True, exist_ok=True)
    # Native controls cover the union of selected modes, once per GPU host.
    tests = [mode for mode in WORKLOADS if any(mode in modes for modes in selection["cases"].values())]
    runner = Runner(inventory, args.ssh_key, args.artifacts, args.results, tests, args.jobs)
    try:
        runner.deploy()
        runner.baseline()
        for name, modes in selection["cases"].items():
            runner.topology(name, matrix["topologies"][name], modes)
    except Exception as error:
        runner.record("fleet.setup", subprocess.CompletedProcess([], 1, str(error)))
    return 1 if runner.report() else 0


if __name__ == "__main__":
    raise SystemExit(main())
