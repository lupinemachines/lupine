#!/usr/bin/env python3
"""Run the graph lifetime regression and bound the server child's RSS growth.

The test prints memory-step=N and waits for a newline after each empty capture.
Supply the dedicated server's listener pidfile; --ssh observes a remote server.
"""

import argparse
import os
import selectors
import shlex
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh")
    parser.add_argument("--pid-file", required=True)
    parser.add_argument("--max-growth-kib", type=int, default=32768)
    parser.add_argument("client", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.client
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("a client command is required")

    # Each integration server forks exactly one child for this client.
    query = f'ps -o rss= --ppid "$(cat {shlex.quote(args.pid_file)})"'
    observer = ["ssh", "-o", "BatchMode=yes", args.ssh, query] if args.ssh else ["sh", "-c", query]
    samples = []
    with subprocess.Popen(command + ["--memory-steps"], stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as client:
        selector = selectors.DefaultSelector()
        selector.register(client.stdout, selectors.EVENT_READ)
        try:
            output = b""
            while True:
                if not selector.select(timeout=60):
                    raise RuntimeError("graph lifetime regression timed out")
                chunk = os.read(client.stdout.fileno(), 4096)
                if not chunk:
                    break
                output += chunk
                while b"\n" in output:
                    line, output = output.split(b"\n", 1)
                    print(line.decode(errors="replace"), flush=True)
                    if not line.startswith(b"memory-step="):
                        continue
                    # Native user-object destruction is asynchronous even after
                    # stream synchronization. Allow the cleanup worker to drain.
                    deadline = time.monotonic() + 5
                    while True:
                        time.sleep(0.1)
                        rows = subprocess.check_output(observer, text=True, timeout=10).split()
                        if len(rows) != 1:
                            raise RuntimeError(f"expected one server child, got {rows}")
                        rss = int(rows[0])
                        if not samples or rss <= samples[0] + args.max_growth_kib:
                            break
                        if time.monotonic() >= deadline:
                            raise AssertionError(f"server RSS grew from {samples[0]} to {rss} KiB")
                    samples.append(rss)
                    print(f"server RSS: {rss} KiB", flush=True)
                    client.stdin.write(b"\n")
                    client.stdin.flush()
            if client.wait(timeout=10) != 0:
                raise RuntimeError("graph lifetime regression failed")
            if len(samples) != 9:
                raise RuntimeError(f"expected nine RSS samples, got {len(samples)}")
            print(f"PASS: server RSS growth {max(samples) - samples[0]} KiB")
        finally:
            selector.close()
            if client.poll() is None:
                client.kill()
                client.wait()


if __name__ == "__main__":
    main()
