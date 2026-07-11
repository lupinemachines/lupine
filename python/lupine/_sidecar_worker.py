import ctypes
import json
import math
import sys
import traceback

import torch

objects = {}
next_handle = 1
TENSOR_CHUNK_BYTES = 8 * 1024 * 1024


class TensorStreamError(RuntimeError):
    pass


def dtype_name(dtype):
    return str(dtype).removeprefix("torch.")


def tensor_nbytes(tensor):
    return int(tensor.numel()) * int(tensor.element_size())


def tensor_bytes(tensor):
    size = tensor_nbytes(tensor)
    if size == 0:
        return memoryview(b"")
    storage = (ctypes.c_ubyte * size).from_address(tensor.data_ptr())
    return memoryview(storage).cast("B")


def write_all(stream, data):
    view = memoryview(data).cast("B")
    offset = 0
    while offset < len(view):
        written = stream.write(view[offset:])
        if written is None or written <= 0:
            raise RuntimeError("stream stopped accepting data")
        offset += written


def write_tensor(stream, tensor):
    data = tensor_bytes(tensor)
    for offset in range(0, len(data), TENSOR_CHUNK_BYTES):
        write_all(stream, data[offset : offset + TENSOR_CHUNK_BYTES])


def read_tensor(stream, tensor):
    data = tensor_bytes(tensor)
    offset = 0
    while offset < len(data):
        end = min(offset + TENSOR_CHUNK_BYTES, len(data))
        read = stream.readinto(data[offset:end])
        if read is None or read <= 0:
            raise TensorStreamError(
                "tensor stream ended before the tensor was complete"
            )
        offset += read


def discard_bytes(stream, size):
    buffer = bytearray(min(size, TENSOR_CHUNK_BYTES))
    view = memoryview(buffer)
    remaining = size
    while remaining:
        count = min(remaining, len(view))
        read = stream.readinto(view[:count])
        if read is None or read <= 0:
            raise TensorStreamError(
                "tensor stream ended before the tensor was complete"
            )
        remaining -= read


def output_tensor_metadata(output):
    tensor, dtype = output
    element_size = torch.empty((), dtype=dtype).element_size()
    return {
        "shape": list(tensor.shape),
        "dtype": dtype_name(dtype),
        "layout": "strided",
        "device": "cpu",
        "byteorder": sys.byteorder,
        "nbytes": int(tensor.numel()) * int(element_size),
    }


def encode_cpu_tensor(tensor, tensors):
    if tensor.device.type != "cpu":
        raise ValueError(f"cannot return a tensor on {tensor.device} from the sidecar")
    if tensor.layout != torch.strided or tensor.is_quantized:
        raise ValueError("sidecar CPU tensor results require unquantized strided layout")
    if tensor.is_conj() or tensor.is_neg():
        raise ValueError("sidecar CPU tensor results cannot be lazy views")
    if not tensor.is_contiguous():
        raise ValueError("sidecar CPU tensor results must be contiguous")
    tensor = tensor.detach()
    stream_index = len(tensors)
    tensors.append((tensor, tensor.dtype))
    return {
        "type": "tensor_data",
        "stream": stream_index,
    }


def parse_tensor_metadata(payload):
    if payload.get("device") != "cpu" or payload.get("layout") != "strided":
        raise TensorStreamError("invalid CPU tensor description")
    if payload.get("byteorder") != sys.byteorder:
        raise TensorStreamError("CPU tensor byte order does not match the worker")
    try:
        shape = tuple(int(dimension) for dimension in payload["shape"])
        dtype = getattr(torch, payload["dtype"])
        size = int(payload["nbytes"])
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise TensorStreamError("malformed CPU tensor description") from exc
    if any(dimension < 0 for dimension in shape):
        raise TensorStreamError("CPU tensor shape contains a negative dimension")
    element_count = math.prod(shape)
    element_size = torch.empty((), dtype=dtype).element_size()
    expected_bytes = element_count * element_size
    if size != expected_bytes:
        raise TensorStreamError(
            f"wrong CPU tensor byte count: expected {expected_bytes}, got {size}"
        )
    return shape, dtype, element_count, element_size


def decode_cpu_tensor(payload, stream):
    shape, dtype, _, _ = parse_tensor_metadata(payload)
    try:
        tensor = torch.empty(shape, dtype=dtype)
    except Exception:
        discard_bytes(stream, int(payload["nbytes"]))
        raise
    read_tensor(stream, tensor)
    return tensor


def read_cuda_tensor(stream, payload, *, dtype=None, device="cuda:0", destination=None):
    shape, source_dtype, element_count, element_size = parse_tensor_metadata(payload)
    size = int(payload["nbytes"])
    consumed = 0
    try:
        direct = (
            destination is not None
            and tuple(destination.shape) == shape
            and destination.is_contiguous()
        )
        if direct:
            tensor = destination
        else:
            target_dtype = source_dtype if dtype is None else dtype
            tensor = torch.empty(shape, dtype=target_dtype, device=device)

        flat = tensor.reshape(-1)
        chunk_elements = max(1, TENSOR_CHUNK_BYTES // element_size)
        staging = bytearray(min(size, chunk_elements * element_size))
        staging_view = memoryview(staging)
        for element_offset in range(0, element_count, chunk_elements):
            count = min(chunk_elements, element_count - element_offset)
            chunk_bytes = count * element_size
            offset = 0
            while offset < chunk_bytes:
                read = stream.readinto(staging_view[offset:chunk_bytes])
                if read is None or read <= 0:
                    raise TensorStreamError(
                        "tensor stream ended before the tensor was complete"
                    )
                offset += read
            consumed += chunk_bytes
            source = torch.frombuffer(staging, dtype=source_dtype, count=count)
            flat[element_offset : element_offset + count].copy_(source)

        if destination is not None and not direct:
            destination.copy_(tensor)
            return destination
        return tensor
    except TensorStreamError:
        raise
    except Exception:
        discard_bytes(stream, size - consumed)
        raise


def write_cuda_tensor(stream, tensor, dtype):
    source = tensor.detach().reshape(-1)
    element_size = torch.empty((), dtype=dtype).element_size()
    chunk_elements = max(1, TENSOR_CHUNK_BYTES // element_size)
    staging = torch.empty(
        min(source.numel(), chunk_elements), dtype=dtype, device="cpu"
    )
    for offset in range(0, source.numel(), chunk_elements):
        count = min(chunk_elements, source.numel() - offset)
        chunk = staging[:count]
        chunk.copy_(source[offset : offset + count])
        write_tensor(stream, chunk)


def store(tensor, tensors):
    global next_handle
    if tensor.device.type == "cpu":
        return encode_cpu_tensor(tensor, tensors)
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
            tensors.append(decode_cpu_tensor(description, stream))
        except TensorStreamError:
            raise
        except Exception:
            for remaining in descriptions[index + 1 :]:
                discard_bytes(stream, int(remaining["nbytes"]))
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
            discard_bytes(input_stream, int(descriptions[0]["nbytes"]))
            raise
        tensor = read_cuda_tensor(
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
            discard_bytes(input_stream, int(descriptions[0]["nbytes"]))
            raise
        read_cuda_tensor(
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
                    torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else None
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
                parse_tensor_metadata(description)
            if request.get("op") == "upload":
                if len(descriptions) != 1:
                    raise TensorStreamError(
                        "upload requires exactly one tensor stream"
                    )
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
            output_tensor_metadata(output) for output in output_tensors
        ]
        write_all(output_stream, json.dumps(response).encode("utf-8") + b"\n")
        for tensor, dtype in output_tensors:
            if tensor.device.type == "cpu":
                write_tensor(output_stream, tensor)
            else:
                write_cuda_tensor(output_stream, tensor, dtype)
        output_stream.flush()
        output_tensors.clear()


if __name__ == "__main__":
    main()
