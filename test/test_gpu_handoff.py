import os
from pathlib import Path
import select
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid


def line(process):
    data = b""
    deadline = time.monotonic() + 30
    while not data.endswith(b"\n"):
        remaining = max(0, deadline - time.monotonic())
        if not select.select([process.stdout], [], [], remaining)[0]:
            raise RuntimeError("client stalled during handoff")
        byte = os.read(process.stdout.fileno(), 1)
        if not byte:
            raise RuntimeError(f"client exited: {process.poll()}")
        data += byte
    text = data.decode().strip()
    print(text, flush=True)
    return text


def main():
    server_binary, client_binary, client_library = sys.argv[1:]
    provider = os.environ.get("LUPINE_CHECKPOINT_LIBRARY", "")
    if not provider or not Path(provider).is_file():
        return 77
    try:
        subprocess.run(["nvidia-smi", "-L"], check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return 77
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    processes = []
    with tempfile.TemporaryDirectory(prefix="lupine-handoff-") as directory:
        env = dict(os.environ, LUPINE_PORT=str(port),
                   LUPINECR_CHECKPOINT_DIR=directory, LUPINE_CHECKPOINT_OBJECTS="1")

        def start_server():
            server = subprocess.Popen([server_binary], env=env, start_new_session=True)
            processes.append(server)
            # A TCP-only probe creates no CUDA state in its short-lived child.
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                        return server
                except OSError:
                    if server.poll() is not None:
                        raise RuntimeError("server exited before listening")
                    time.sleep(0.02)
            raise RuntimeError("server did not listen")

        try:
            server = start_server()
            client_env = dict(os.environ, LUPINE_SERVER=f"127.0.0.1:{port}",
                              LUPINE_SESSION=str(uuid.uuid4()), LD_LIBRARY_PATH=client_library)
            client_env.pop("CUDA_VISIBLE_DEVICES", None)
            client = subprocess.Popen([client_binary], env=client_env, stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, text=True, bufsize=1, start_new_session=True)
            processes.append(client)
            for cycle in range(3):
                assert line(client) == f"READY {cycle}"
                server.send_signal(signal.SIGTERM)
                assert server.wait(timeout=30) == 0
                client.stdin.write("\n")
                client.stdin.flush()
                if os.environ.get("LUPINE_TEST_REJECT"):
                    assert line(client) == "REJECTED"
                    assert not list(Path(directory).glob("manifests/*.manifest"))
                    assert client.wait(timeout=10) == 0
                    return 0
                # Older generations no longer pin GPU chunks or module images.
                assert len(list(Path(directory).glob("manifests/*.manifest"))) == 1
                assert len(list(Path(directory).glob("manifests/*.manifest.cuda"))) == 1
                # The application has entered its next CUDA call, but capacity
                # has not returned yet. It must remain alive and blocked.
                assert not select.select([client.stdout], [], [], 0.2)[0]
                server = start_server()
                assert line(client) == f"RESTORED {cycle} {43 + cycle}"
            assert client.wait(timeout=10) == 0
            return 0
        finally:
            for process in reversed(processes):
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


if __name__ == "__main__":
    sys.exit(main())
