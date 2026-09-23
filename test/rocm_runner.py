#!/usr/bin/env python3
"""Opt-in upstream ROCm runners; see rocm/README.md for environment options."""

import argparse
import contextlib
import dataclasses
import datetime
import json
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parent.parent


def words(value):
    return shlex.split(value)


def native_environment(rocm):
    # Builds, test discovery and servers must never inherit a client preload.
    env = dict(os.environ)
    for key in ("LD_PRELOAD", "LD_LIBRARY_PATH", "LUPINE_SERVER", "LD_BIND_NOW"):
        env.pop(key, None)
    env["LD_LIBRARY_PATH"] = os.environ.get("ROCM_LD_LIBRARY_PATH", str(rocm / "lib"))
    env["PATH"] = f"{rocm / 'bin'}:{rocm / 'llvm/bin'}:{env['PATH']}"
    env["HIP_PLATFORM"] = "amd"
    return env


def checked(command, **kwargs):
    return subprocess.check_output(command, text=True, **kwargs).strip()


def checkout(url, ref, source, libraries):
    if not source.exists():
        source.mkdir(parents=True)
        subprocess.run(["git", "init", "-q", str(source)], check=True)
        subprocess.run(["git", "-C", str(source), "remote", "add", "origin", url], check=True)
        if libraries:
            subprocess.run(["git", "-C", str(source), "sparse-checkout", "set",
                            "--cone", "shared", *[f"projects/{lib}" for lib in libraries]], check=True)
    git = ["git", "-C", str(source)]
    current = subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True)
    wanted = subprocess.run(git + ["rev-parse", f"{ref}^{{commit}}"], capture_output=True, text=True)
    if current.returncode == wanted.returncode == 0 and current.stdout == wanted.stdout:
        return
    if checked(git + ["status", "--porcelain", "--untracked-files=no"]):
        raise RuntimeError(f"refusing to change a modified checkout: {source}")
    subprocess.run(git + ["fetch", "--quiet", "--depth=1", "--filter=blob:none", "origin", ref], check=True)
    subprocess.run(git + ["-c", "advice.detachedHead=false", "checkout", "--quiet", "--detach", "FETCH_HEAD"], check=True)


def leaf_projects(source, roots):
    projects = set()
    for root in roots:
        directory = source / root
        if not directory.is_dir():
            raise RuntimeError(f"missing suite directory: {directory}")
        for cmake in directory.rglob("CMakeLists.txt"):
            relative = cmake.parent.relative_to(source)
            if not {"cmake", "CMakeFiles", "build", "common", "Common", "External"}.intersection(relative.parts):
                projects.add(relative.as_posix())
    return sorted(project for project in projects
                  if not any(other.startswith(project + "/") for other in projects))


def cmake_targets(build):
    """Read actual executable artifacts, including targets excluded from ALL."""
    reply = build / ".cmake/api/v1/reply"
    indexes = sorted(reply.glob("index-*.json"))
    if not indexes:
        raise RuntimeError(f"missing CMake target metadata in {build}; run with building enabled first")
    index = json.loads(indexes[-1].read_text())
    model = json.loads((reply / index["reply"]["codemodel-v2"]["jsonFile"]).read_text())
    targets = []
    for target in model["configurations"][0]["targets"]:
        detail = json.loads((reply / target["jsonFile"]).read_text())
        if detail["type"] == "EXECUTABLE":
            targets.append(detail)
    return targets


@dataclasses.dataclass
class Unit:
    name: str
    cwd: Path
    command: list
    status: str = ""
    detail: str = ""
    executables: list = dataclasses.field(default_factory=list)


class Suite:
    def __init__(self, name):
        self.name = name
        self.prefix = name.upper().replace("-", "_")
        self.source = Path(self.option("DIR", str(ROOT / "test" / name / name))).resolve()
        self.build = Path(self.option("BUILD_DIR", str(self.source / "build"))).resolve()
        self.ref = self.option("REF", "rocm-7.2.0")
        self.rocm = Path(os.environ.get("ROCM_PATH", "/opt/rocm")).resolve()
        self.env = native_environment(self.rocm)
        self.build_mode = os.environ.get("BUILD_SAMPLES" if name == "rocm-examples" else "BUILD_TESTS", "auto")
        if self.build_mode not in ("auto", "0", "1"):
            raise RuntimeError("BUILD_TESTS / BUILD_SAMPLES must be auto, 0 or 1")
        self.jobs = os.environ.get("JOBS", str(min(os.cpu_count() or 1, 8)))
        self.libs = words(self.option("LIBS", "rocblas hipblas")) if name == "rocm-libraries" else []
        if any(lib not in ("rocblas", "hipblas") for lib in self.libs):
            raise RuntimeError("ROCM_LIBRARIES_LIBS currently supports rocblas and hipblas")
        self.configured = set()
        self.arch = os.environ.get("ROCM_ARCH", "")

    def option(self, name, default=""):
        return os.environ.get(f"{self.prefix}_{name}", default)

    def prepare(self):
        if self.build_mode != "0":
            checkout(self.option("URL", f"https://github.com/ROCm/{self.name}.git"),
                     self.ref, self.source, self.libs)
            if self.libs and any(not (self.source / "projects" / lib).is_dir() for lib in self.libs):
                sparse = subprocess.run(["git", "-C", str(self.source), "config", "--bool", "core.sparseCheckout"],
                                        capture_output=True, text=True)
                if sparse.stdout.strip() == "true":
                    subprocess.run(["git", "-C", str(self.source), "sparse-checkout", "add",
                                    *[f"projects/{lib}" for lib in self.libs]], check=True)
        if not self.source.is_dir():
            raise RuntimeError(f"missing checkout: {self.source}")

    def projects(self):
        if self.name == "hip-tests":
            catch = sorted(path.parent.relative_to(self.source).as_posix()
                           for path in (self.source / "catch/unit").glob("*/CMakeLists.txt"))
            return catch + leaf_projects(self.source, ["samples"])
        if self.name == "rocm-examples":
            return leaf_projects(self.source, words(self.option("ROOTS", "HIP-Basic Libraries Applications")))
        return [f"{lib}/{lib}-{kind}" for lib in self.libs for kind in ("test", "bench")]

    def architecture(self):
        if not self.arch:
            output = checked([str(self.rocm / "bin/rocm_agent_enumerator")], env=self.env)
            self.arch = ";".join(sorted(set(re.findall(r"^gfx(?!000)[0-9a-f]+(?::\S+)?$", output, re.M))))
        if not self.arch:
            raise RuntimeError("no AMD GPU detected; set ROCM_ARCH to the server's gfx architecture")
        return self.arch

    def configure(self, source, build, extra, log):
        if build in self.configured:
            return
        query = build / ".cmake/api/v1/query/codemodel-v2"
        query.parent.mkdir(parents=True, exist_ok=True)
        query.touch()
        args = ["cmake", "-S", str(source), "-B", str(build), "-G", "Ninja",
                "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_PREFIX_PATH={self.rocm}",
                f"-DROCM_PATH={self.rocm}", f"-DROCM_ROOT={self.rocm}",
                f"-DCMAKE_HIP_ARCHITECTURES={self.architecture()}",
                f"-DGPU_TARGETS={self.architecture()}", f"-DAMDGPU_TARGETS={self.architecture()}",
                f"-DCMAKE_CXX_COMPILER={self.rocm / 'bin/hipcc'}",
                *extra, *words(self.option("CMAKE_ARGS"))]
        # Older hip-tests samples compile device code as CXX, not HIP.
        self.env["HIPCC_COMPILE_FLAGS_APPEND"] = " ".join(
            f"--offload-arch={arch}" for arch in self.architecture().split(";"))
        subprocess.run(args, check=True, stdout=log, stderr=subprocess.STDOUT, env=self.env)
        self.configured.add(build)

    def units(self, project):
        is_catch = project.startswith("catch/") and self.name == "hip-tests"
        library = project.split("/")[0]
        if is_catch:
            source, build = self.source / "catch", self.build / "catch"
            extra = []
            if self.build_mode != "0":
                offload = " ".join(f"--offload-arch={arch}" for arch in self.architecture().split(";"))
                extra = ["-DHIP_PLATFORM=amd", f"-DHIP_PATH={self.rocm}", f"-DOFFLOAD_ARCH_STR={offload}"]
        elif self.name == "rocm-libraries":
            source, build = self.source / "projects" / library, self.build / library
            extra = ["-DSKIP_LIBRARY=ON", "-DBUILD_CLIENTS_TESTS=ON", "-DBUILD_CLIENTS_BENCHMARKS=ON",
                     "-DBUILD_FORTRAN_CLIENTS=OFF", "-DLINK_BLIS=OFF",
                     "-DBLA_SIZEOF_INTEGER=8", "-DBLA_VENDOR=OpenBLAS", "-DBLA_STATIC=OFF",
                     f"-D{library.upper()}_LIBRARY_DIR={self.rocm}"]
            # FindBLAS finds the library but does not expose CBLAS headers.
            # In particular Ubuntu installs ILP64 headers outside the default
            # include path, alongside a potentially incompatible LP64 cblas.h.
            if self.build_mode != "0":
                for variable, option in (("includedir", "BLAS_INCLUDE_DIR"), ("libdir", "CMAKE_LIBRARY_PATH")):
                    result = subprocess.run(["pkg-config", f"--variable={variable}", "openblas64"],
                                            capture_output=True, text=True, env=self.env)
                    if result.returncode == 0 and result.stdout.strip():
                        extra.append(f"-D{option}={result.stdout.strip()}")
        else:
            source, build, extra = self.source / project, self.build / project, []
        log_path = self.build / f"{project}.build.log"
        try:
            if self.build_mode != "0":
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("w") as log:
                    self.configure(source, build, extra, log)
                    targets = self.select_targets(build, project, is_catch)
                    if not targets:
                        raise RuntimeError("no executable CMake targets found")
                    subprocess.run(["cmake", "--build", str(build), "--parallel", self.jobs,
                                    "--target", *[target["name"] for target in targets]],
                                   check=True, stdout=log, stderr=subprocess.STDOUT, env=self.env)
            targets = self.select_targets(build, project, is_catch)
            if not targets or any(not (build / target["artifacts"][0]["path"]).is_file() for target in targets):
                raise RuntimeError("selected executables are not built")
        except (subprocess.CalledProcessError, RuntimeError) as error:
            return [Unit(project, build, [], "SKIP:build-failed", f"{error}; see {log_path}")]
        if is_catch:
            directory = build / "catch_tests" / project.removeprefix("catch/")
            return [Unit(project, directory, ["ctest", "--output-on-failure", "--no-tests=error",
                                             *words(self.option("CTEST_ARGS"))],
                         executables=[build / target["artifacts"][0]["path"] for target in targets])]
        units = []
        for target in targets:
            exe = build / target["artifacts"][0]["path"]
            args = []
            if self.name == "rocm-libraries":
                if target["name"].endswith("-test"):
                    args = ["--gtest_filter=" + self.option("GTEST_FILTER", "quick*"), "--gtest_color=no"]
                else:
                    args = words(self.option("BENCH_ARGS", "-f gemm -r f32_r -m 32 -n 32 -k 32 -i 1 -j 0 -v 1"))
                name = project
            else:
                name = f"{project}/{target['name']}"
            units.append(Unit(name, exe.parent, [str(exe), *args], executables=[exe]))
        return units

    def select_targets(self, build, project, is_catch):
        targets = cmake_targets(build)
        if is_catch:
            directory = project.removeprefix("catch/")
            targets = [target for target in targets
                       if target["paths"]["source"] == directory
                       or target["paths"]["source"].startswith(directory + "/")]
        elif self.name == "rocm-libraries":
            targets = [target for target in targets if target["name"] == project.split("/")[1]]
        return sorted(targets, key=lambda target: target["name"])


def stop_process(process):
    # Kill the process group too: tests and servers can fork children.
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        # The grace period expired; force-kill the process group below.
        pass
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    process.wait()


class Server:
    def __init__(self, env):
        self.env = dict(env)
        self.env["LD_LIBRARY_PATH"] = os.environ.get("SERVER_LD_LIBRARY_PATH", env["LD_LIBRARY_PATH"])
        self.local_bin = Path(os.environ.get("SERVER_LOCAL_BIN", str(ROOT / "build/lupine_driver_server"))).resolve()
        self.mode = os.environ.get("SERVER_MODE", "local")
        if self.mode not in ("local", "ssh"):
            raise RuntimeError("SERVER_MODE must be local or ssh")
        self.host = os.environ.get("SERVER_HOST", "127.0.0.1")
        self.target = os.environ.get("SERVER_SSH_TARGET", self.host)
        self.ssh = ["ssh", *words(os.environ.get("SSH_OPTS", "")), self.target]
        self.ssh_timeout = float(os.environ.get("SSH_COMMAND_TIMEOUT", "45"))
        self.remote_bin = os.environ.get("SERVER_REMOTE_BIN", f"/tmp/lupine-rocm-{os.getpid()}")
        self.uploaded = False

    def remote(self, command):
        return checked(self.ssh + [command], env=self.env, timeout=self.ssh_timeout)

    def upload(self):
        if self.mode == "ssh" and os.environ.get("SERVER_UPLOAD", "1") == "1":
            subprocess.run(["scp", *words(os.environ.get("SSH_OPTS", "")), str(self.local_bin),
                            f"{self.target}:{self.remote_bin}"], check=True, env=self.env, timeout=self.ssh_timeout)
            self.uploaded = True

    def cleanup(self):
        if self.uploaded and os.environ.get("SERVER_REMOTE_CLEANUP", "1") == "1":
            self.remote("rm -f -- " + shlex.quote(self.remote_bin))

    @contextlib.contextmanager
    def running(self, port, log_path):
        try:
            with socket.create_connection((self.host, port), timeout=0.2):
                raise RuntimeError(f"server port already in use: {self.host}:{port}")
        except OSError:
            pass
        process = None
        remote_pid = None
        remote_log = f"{self.remote_bin}-{port}.log"
        with log_path.open("w") as log:
            try:
                if self.mode == "local":
                    process = subprocess.Popen([str(self.local_bin)], env={**self.env, "LUPINE_PORT": str(port)},
                                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                else:
                    command = shlex.join(["env", "-u", "LD_PRELOAD", "-u", "LUPINE_SERVER",
                                          f"LD_LIBRARY_PATH={self.env['LD_LIBRARY_PATH']}", f"LUPINE_PORT={port}",
                                          "setsid", self.remote_bin])
                    remote_pid = self.remote(f"nohup {command} >{shlex.quote(remote_log)} 2>&1 </dev/null & echo $!")
                    if not remote_pid.isdigit():
                        raise RuntimeError("SSH did not return a server PID")
                deadline = time.monotonic() + 10
                while True:
                    if process is not None and process.poll() is not None:
                        raise RuntimeError(f"server exited; see {log_path}")
                    try:
                        with socket.create_connection((self.host, port), timeout=0.2):
                            break
                    except OSError:
                        if time.monotonic() >= deadline:
                            raise RuntimeError(f"server did not listen on {self.host}:{port}")
                        time.sleep(0.1)
                yield
            finally:
                if process is not None:
                    stop_process(process)
                if remote_pid is not None and remote_pid.isdigit():
                    self.remote(f"kill -TERM -{remote_pid} 2>/dev/null || true; "
                                f"sleep 0.2; kill -KILL -{remote_pid} 2>/dev/null || true")
                    log.write(self.remote(f"cat {shlex.quote(remote_log)}; rm -f -- {shlex.quote(remote_log)}"))


def shim_environment(env, directory):
    shim = Path(os.environ.get("LUPINE_LIB", str(ROOT / "build/libamdhip64.so.1"))).resolve(strict=True)
    # The current shim's SONAME is .1; real ROCm programs need .6 or .7.
    # Resolve every HIP dependency to the shim, so unimplemented symbols fail
    # visibly instead of falling through to the local GPU (issue #659).
    for name in ("libamdhip64.so", "libamdhip64.so.1", "libamdhip64.so.6", "libamdhip64.so.7"):
        (directory / name).symlink_to(shim)
    return {**env, "LD_PRELOAD": str(shim), "LD_BIND_NOW": "1",
            "LD_LIBRARY_PATH": f"{directory}:{env['LD_LIBRARY_PATH']}"}


def verify_loader(unit, env):
    # Legacy DT_RPATH can take precedence over LD_LIBRARY_PATH. Check the
    # actual executable dependencies as well as staging SONAME aliases.
    shim = Path(env["LD_PRELOAD"]).resolve()
    for executable in unit.executables:
        output = checked(["ldd", str(executable)], env=env, stderr=subprocess.DEVNULL, timeout=10)
        for line in output.splitlines():
            if "libamdhip64.so" not in line:
                continue
            if "=>" not in line and not re.search(r" \(0x[0-9a-f]+\)$", line):
                continue  # Loader version warnings are not dependency entries.
            path = line.split("=>", 1)[-1].rsplit(" (", 1)[0].strip()
            if Path(path).resolve() != shim:
                raise RuntimeError(f"{executable} resolves HIP to {path}, not the Lupine shim")


def execute(unit, env, timeout, log_path):
    with log_path.open("w") as log:
        log.write(f"cwd: {unit.cwd}\ncommand: {shlex.join(unit.command)}\n")
        log.flush()
        process = subprocess.Popen(unit.command, cwd=unit.cwd, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            log.write(f"\ntimed out after {timeout}s\n")
            return "FAIL:124"
        finally:
            stop_process(process)
    output = log_path.read_text(errors="replace")
    if rc:
        return f"FAIL:{rc}"
    if re.search(r"\bFAILED\b|Validation failed|Running 0 tests|No tests were found", output):
        return "FAIL:output"
    return "PASS"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=("hip-tests", "rocm-examples", "rocm-libraries"))
    parser.add_argument("projects", nargs="*", help="feature/sample directories, or library/test executable names")
    args = parser.parse_args()
    suite = Suite(args.suite)
    suite.prepare()
    projects = suite.projects()
    if args.projects:
        selected = []
        for project in args.projects:
            matches = [name for name in projects if name == project or name.startswith(project.rstrip("/") + "/")]
            if not matches:
                raise RuntimeError(f"unknown project {project!r}; use LIST_TESTS=1 to list available projects")
            selected.extend(matches)
        projects = list(dict.fromkeys(selected))
    if not projects:
        raise RuntimeError("no projects selected")
    if os.environ.get("LIST_TESTS") == "1":
        print("\n".join(projects))
        return 0
    units = []
    for project in projects:
        print(f"preparing {suite.name}: {project}", file=sys.stderr, flush=True)
        units.extend(suite.units(project))
    if os.environ.get("BUILD_ONLY") == "1":
        for unit in units:
            if unit.status:
                print(f"{unit.name}: {unit.status}: {unit.detail}", file=sys.stderr)
        return int(any(unit.status for unit in units))

    native = os.environ.get("RUN_NATIVE", "0") == "1"
    known_path = Path(suite.option("KNOWN_FAILURES", str(ROOT / "test" / suite.name / "known_failures.txt")))
    known = set()
    if known_path.exists() and not native:
        known = {line.split("#", 1)[0].strip() for line in known_path.read_text().splitlines()}
    disabled = set(suite.option("SKIP_LIST").replace(",", " ").split())
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    results = Path(os.environ.get("RESULTS_DIR", str(ROOT / "test" / suite.name / "results" / timestamp))).resolve()
    results.mkdir(parents=True, exist_ok=True)
    timeout = float(os.environ.get("TEST_TIMEOUT", os.environ.get("SAMPLE_TIMEOUT", "300")))
    port_base = int(os.environ.get("SERVER_PORT_BASE", "23000"))
    if not 0 < port_base <= port_base + len(units) - 1 <= 65535:
        raise RuntimeError("per-unit ports must be in 1..65535")
    if len(units) > int(os.environ.get("SERVER_PORT_COUNT", "65535")):
        raise RuntimeError("selected units exceed SERVER_PORT_COUNT; reserve a larger port range")
    for unit in units:
        if unit.status:
            continue
        if unit.name in disabled:
            unit.status, unit.detail = "SKIP:disabled", "disabled by skip list"
        elif unit.name in known:
            unit.status, unit.detail = "SKIP:known", f"listed in {known_path}"
    server = Server(suite.env)
    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0}
    build_failed = any(unit.status == "SKIP:build-failed" for unit in units)
    with tempfile.TemporaryDirectory(prefix="lupine-hip-shim-") as shim_dir, (results / "results.tsv").open("w") as tsv:
        env = suite.env
        try:
            if not native and any(not unit.status for unit in units):
                env = shim_environment(env, Path(shim_dir))
                server.upload()
            for index, unit in enumerate(units):
                status, detail = unit.status, unit.detail
                log_path = results / f"{unit.name}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                if not status:
                    print(f"[{index + 1}/{len(units)}] {unit.name} ({'native' if native else 'lupine'})", file=sys.stderr, flush=True)
                    try:
                        server_context = contextlib.nullcontext() if native else server.running(
                            port_base + index, log_path.with_suffix(".server.log"))
                        with server_context:
                            client_env = env if native else {**env, "LUPINE_SERVER": f"{server.host}:{port_base + index}"}
                            if not native:
                                verify_loader(unit, client_env)
                            status = execute(unit, client_env, timeout, log_path)
                    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                        status, detail = "FAIL:runner", str(error)
                    if not detail:
                        detail = str(log_path)
                counts[status.split(":", 1)[0]] += 1
                row = f"{unit.name}\t{status}\t{' '.join(detail.split())}"
                print(row, flush=True)
                tsv.write(row + "\n")
                tsv.flush()
        finally:
            server.cleanup()
    summary = "\n".join(f"{key} {value}" for key, value in counts.items())
    summary += f"\nTOTAL {len(units)}\nMODE {'native' if native else 'lupine'}\nRESULTS {results / 'results.tsv'}\n"
    (results / "summary.txt").write_text(summary)
    print(summary, end="")
    if counts["FAIL"] or build_failed:
        return 1
    return 77 if counts["PASS"] == 0 else 0


if __name__ == "__main__":
    # Let context managers reap children on Ctrl-C and CTest timeout/SIGTERM.
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    try:
        sys.exit(main())
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
