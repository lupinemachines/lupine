"""Remote PyTorch worker entrypoint for a LUPINE sidecar session."""

import json
import sys
import traceback

import torch

from .tensor import (
    TensorStreamError,
    _decode_cpu_tensor,
    _discard_bytes,
    _parse_tensor_metadata,
    _read_cuda_tensor,
    _tensor_stream_metadata,
    _validate_cpu_result_tensor,
    _write_all,
    _write_cuda_tensor,
    _write_tensor,
)


objects = {}
next_handle = 1


def store(tensor, tensors):
    global next_handle
    if tensor.device.type == "cpu":
        _validate_cpu_result_tensor(tensor)
        tensor = tensor.detach()
        stream_index = len(tensors)
        tensors.append((tensor, tensor.dtype))
        return {
            "type": "tensor_data",
            "stream": stream_index,
        }
    if tensor.device.type != "cuda":
        raise ValueError(f"unsupported sidecar result device: {tensor.device}")
    handle = next_handle
    next_handle += 1
    objects[handle] = tensor
    return {
        "type": "tensor",
        "handle": handle,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
    }


def receive_tensors(stream, descriptions):
    tensors = []
    for index, description in enumerate(descriptions):
        try:
            tensors.append(_decode_cpu_tensor(description, stream))
        except TensorStreamError:
            raise
        except Exception:
            for remaining in descriptions[index + 1 :]:
                _discard_bytes(stream, int(remaining["nbytes"]))
            raise
    return tensors


def decode(value, tensors):
    if isinstance(value, list):
        return [decode(item, tensors) for item in value]
    if isinstance(value, dict) and "__tuple__" in value:
        return tuple(decode(item, tensors) for item in value["__tuple__"])
    if isinstance(value, dict) and "__sidecar_tensor__" in value:
        return objects[int(value["__sidecar_tensor__"])]
    if isinstance(value, dict) and "__cpu_tensor__" in value:
        stream_index = int(value["__cpu_tensor__"])
        if stream_index < 0 or stream_index >= len(tensors):
            raise ValueError(f"invalid CPU tensor stream: {stream_index}")
        return tensors[stream_index]
    if isinstance(value, dict) and "__dtype__" in value:
        return getattr(torch, value["__dtype__"])
    if isinstance(value, dict) and "__device__" in value:
        return torch.device(value["__device__"])
    if isinstance(value, dict) and "__layout__" in value:
        return getattr(torch, value["__layout__"])
    if isinstance(value, dict) and "__memory_format__" in value:
        return getattr(torch, value["__memory_format__"])
    if isinstance(value, dict):
        return {key: decode(item, tensors) for key, item in value.items()}
    return value


def encode(value, tensors):
    if isinstance(value, torch.Tensor):
        return store(value, tensors)
    if isinstance(value, torch.Size):
        return {"type": "tuple", "items": list(value)}
    if isinstance(value, tuple):
        return {"type": "tuple", "items": [encode(item, tensors) for item in value]}
    if isinstance(value, list):
        return {"type": "list", "items": [encode(item, tensors) for item in value]}
    if isinstance(value, dict):
        return {
            "type": "dict",
            "items": {key: encode(item, tensors) for key, item in value.items()},
        }
    return {"type": "value", "value": value}


def resolve(packet, overload):
    overload_packet = getattr(torch.ops.aten, packet)
    if overload == "default":
        return overload_packet.default
    return getattr(overload_packet, overload)


def release(value):
    if isinstance(value, dict) and "__sidecar_tensor__" in value:
        objects.pop(int(value["__sidecar_tensor__"]), None)
        return
    if isinstance(value, list):
        for item in value:
            release(item)
        return
    if isinstance(value, dict):
        for item in value.values():
            release(item)


def handle(request, input_stream, descriptions, output_tensors):
    op = request["op"]
    if op == "upload":
        try:
            dtype = getattr(torch, request["dtype"])
            device = request["device"]
        except Exception:
            _discard_bytes(input_stream, int(descriptions[0]["nbytes"]))
            raise
        tensor = _read_cuda_tensor(
            input_stream,
            descriptions[0],
            dtype=dtype,
            device=device,
        )
        return store(tensor, output_tensors)
    if op == "copy_from_cpu":
        try:
            destination = objects[int(request["handle"])]
        except Exception:
            _discard_bytes(input_stream, int(descriptions[0]["nbytes"]))
            raise
        _read_cuda_tensor(
            input_stream,
            descriptions[0],
            destination=destination,
        )
        return True

    input_tensors = receive_tensors(input_stream, descriptions)
    try:
        if op == "ping":
            return {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "device_count": torch.cuda.device_count(),
                "gpu": (
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
                ),
            }
        if op == "call":
            args = decode(request.get("args", []), input_tensors)
            kwargs = decode(request.get("kwargs", {}), input_tensors)
            func = resolve(request["packet"], request["overload"])
            return encode(func(*args, **kwargs), output_tensors)
        if op == "download":
            tensor = objects[int(request["handle"])]
            if tensor.device.type != "cuda":
                raise ValueError("only CUDA tensors can be downloaded")
            if not tensor.is_contiguous():
                raise ValueError("sidecar CUDA tensor downloads must be contiguous")
            dtype = getattr(torch, request["dtype"])
            stream_index = len(output_tensors)
            output_tensors.append((tensor, dtype))
            return {"type": "tensor_data", "stream": stream_index}
        if op == "release":
            release(request["value"])
            return True
        raise RuntimeError(f"unknown op: {op}")
    finally:
        input_tensors.clear()


def main():
    input_stream = sys.stdin.buffer
    output_stream = sys.stdout.buffer
    while True:
        line = input_stream.readline()
        if not line:
            break
        output_tensors = []
        try:
            request = json.loads(line)
            descriptions = request.pop("tensor_streams", [])
            for description in descriptions:
                _parse_tensor_metadata(description)
            if request.get("op") == "upload":
                if len(descriptions) != 1:
                    raise TensorStreamError("upload requires exactly one tensor stream")
            elif request.get("op") == "copy_from_cpu" and len(descriptions) != 1:
                raise TensorStreamError(
                    "copy_from_cpu requires exactly one tensor stream"
                )
        except Exception:
            traceback.print_exc(file=sys.stderr)
            break
        try:
            response = {
                "ok": True,
                "result": handle(
                    request,
                    input_stream,
                    descriptions,
                    output_tensors,
                ),
            }
        except TensorStreamError:
            traceback.print_exc(file=sys.stderr)
            break
        except Exception as exc:
            output_tensors = []
            response = {
                "ok": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        response["tensor_streams"] = [
            _tensor_stream_metadata(tensor, dtype) for tensor, dtype in output_tensors
        ]
        _write_all(output_stream, json.dumps(response).encode("utf-8") + b"\n")
        for tensor, dtype in output_tensors:
            if tensor.device.type == "cpu":
                _write_tensor(output_stream, tensor)
            else:
                _write_cuda_tensor(output_stream, tensor, dtype)
        output_stream.flush()
        output_tensors.clear()


if __name__ == "__main__":
    main()
