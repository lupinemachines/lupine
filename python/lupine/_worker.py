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


def info() -> dict[str, Any]:
    import torch

    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
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


# The host's torch.cuda module forwards its queries and setters here (see
# _backend.device): pickled arguments, the host's current device selected,
# a pickled result. Device properties are rebuilt as the host's own class.
def _cuda(name: str, payload: str, device: int) -> str:
    import base64
    import pickle

    import torch

    args, kwargs = pickle.loads(base64.b64decode(payload))
    with torch.cuda.device(device):
        result = getattr(torch.cuda, name)(*args, **kwargs)
    if isinstance(result, torch.cuda._CudaDeviceProperties):
        from ._backend.device import _Properties

        result = _Properties(
            {
                key: value if isinstance(value, (bool, int, float, str)) else str(value)
                for key in dir(result)
                if not key.startswith("_")
                for value in [getattr(result, key)]
            }
        )
    return base64.b64encode(pickle.dumps(result)).decode()


# CUDA graph capture happens here, on the worker's torch: the ops the host
# keeps issuing during capture run on the capture stream (see
# _C.worker_set_stream) and land in the graph.
def _capture_begin(graph_id: int, capture_error_mode: str, pool: Any = None) -> None:
    import torch

    from ._backend import _C

    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph.capture_begin(pool=pool, capture_error_mode=capture_error_mode)
    _graphs[graph_id] = (graph, stream)
    _C.worker_set_stream(stream.stream_id, stream.device_index)


def _capture_end(graph_id: int) -> None:
    import torch

    from ._backend import _C

    graph, stream = _graphs[graph_id]
    _C.worker_set_stream(-1, 0)
    with torch.cuda.stream(stream):
        graph.capture_end()
    torch.cuda.current_stream().wait_stream(stream)


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
            "_cuda": _cuda,
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
