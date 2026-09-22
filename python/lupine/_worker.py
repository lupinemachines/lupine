"""The torch worker: a CUDA torch that executes the backend's operators.

Run as ``lupine-torch-worker`` (or ``python -m lupine._worker``) in an
interpreter whose torch is the same release as the host's. It loads the
client shims against ``LUPINE_SERVER`` (or ``LUPINE_LIBDIR``) before torch
initialises CUDA, then serves one backend connection and exits when the host
disconnects.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
from typing import Any

READY_PREFIX = "LUPINE_WORKER_READY"

# The namespace exec/eval messages from the host run in.
_namespace: dict[str, Any] = {}
_graphs: dict[int, tuple[Any, Any]] = {}
_capture_stream: Any = None


def info() -> dict[str, Any]:
    import torch

    devices = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "name": props.name,
                "major": props.major,
                "minor": props.minor,
                "multi_processor_count": props.multi_processor_count,
                "total_memory": props.total_memory,
                "uuid": str(getattr(props, "uuid", "")),
                "L2_cache_size": getattr(props, "L2_cache_size", 0),
                "max_threads_per_multi_processor": getattr(
                    props, "max_threads_per_multi_processor", 0
                ),
                "warp_size": getattr(props, "warp_size", 32),
                "is_integrated": getattr(props, "is_integrated", 0),
                "is_multi_gpu_board": getattr(props, "is_multi_gpu_board", 0),
                "regs_per_multiprocessor": getattr(props, "regs_per_multiprocessor", 0),
                "gcnArchName": getattr(props, "gcnArchName", ""),
            }
        )
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device_count": len(devices),
        "devices": devices,
    }


# ``python -m lupine._worker`` runs this file as __main__; the worker's C++
# side imports ``lupine._worker``, a second module object, so the namespace
# is prepared on first use rather than from main().
def _exec(code: str) -> None:
    if "torch" not in _namespace:
        _prepare_namespace()
    exec(code, _namespace)


def _eval(code: str) -> str:
    if "torch" not in _namespace:
        _prepare_namespace()
    return json.dumps(eval(code, _namespace))


def _capturing() -> bool:
    return _capture_stream is not None


# CUDA graph capture happens here, on the worker's torch: the ops the host
# keeps issuing during capture run on the capture stream (see
# _C.worker_set_stream) and land in the graph.
def _capture_begin(graph_id: int, capture_error_mode: str) -> None:
    global _capture_stream
    import torch

    from ._backend import _C

    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph.capture_begin(capture_error_mode=capture_error_mode)
    _graphs[graph_id] = (graph, stream)
    _capture_stream = stream
    _C.worker_set_stream(stream.stream_id, stream.device_index)


def _capture_end(graph_id: int) -> None:
    global _capture_stream
    import torch

    from ._backend import _C

    graph, stream = _graphs[graph_id]
    _C.worker_set_stream(-1, 0)
    with torch.cuda.stream(stream):
        graph.capture_end()
    torch.cuda.current_stream().wait_stream(stream)
    _capture_stream = None


def _replay(graph_id: int) -> None:
    _graphs[graph_id][0].replay()


def _reset(graph_id: int) -> None:
    entry = _graphs.pop(graph_id, None)
    if entry is not None:
        entry[0].reset()


def _prepare_namespace() -> None:
    import torch

    _namespace.update(
        {
            "torch": torch,
            "_capture_begin": _capture_begin,
            "_capture_end": _capture_end,
            "_replay": _replay,
            "_reset": _reset,
            "_capturing": _capturing,
        }
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lupine-torch-worker")
    parser.add_argument(
        "--listen",
        default="127.0.0.1:0",
        help="host:port to accept the backend connection on (port 0 picks one)",
    )
    args = parser.parse_args(argv)
    host, _, port = args.listen.rpartition(":")

    from . import load_native

    load_native(missing_ok=False)
    import torch

    torch.cuda.init()
    if torch.cuda.device_count() == 0:
        print("lupine-torch-worker: no CUDA device is visible", file=sys.stderr)
        return 1
    from ._backend import _C

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((host or "127.0.0.1", int(port)))
    listener.listen(1)
    print(f"{READY_PREFIX} {listener.getsockname()[1]} {torch.__version__}", flush=True)
    connection, _ = listener.accept()
    listener.close()
    _C.worker_serve(connection.detach())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
