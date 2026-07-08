# LUPINE: GPU-over-IP

LUPINE is a GPU over IP bridge allowing GPUs on remote machines to be attached
to CPU-only machines.

## Hosted Demo

Connect to a hosted demo server with a T4 attached for free. This might take a while if there's no GPU
currently provisioned, but subsequent requests should be faster.

```
$ docker run --rm \
  -e LUPINE_SERVER=demo.lupinemachines.com:14833 \
  ghcr.io/lupinemachines/lupine-client:cuda-13.1.0-ubuntu24.04 \
  nvidia-smi -L
GPU 0: Tesla T4 (via lupine demo.lupinemachines.com) (UUID: GPU-b80ae1b9-863f-8f91-7c63-d351fabff035)
```

## Mac Demo

LUPINE lets you spin up a container with a virtual GPU, like connecting a Mac to a Linux GPU server.

```sh
% uname -mors 
Darwin 25.5.0 arm64
% uv run https://raw.githubusercontent.com/lupinemachines/lupine/main/python/examples/tensor.py
LUPINE server host: 100.106.167.98  <-- the ip of a machine with the LUPINE server running
LUPINE server port [14833]: 
cuda available: True
device: lupine:0
count: 1
gpu: NVIDIA GeForce RTX 4090
result: [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]
```

## Quick Start

Use the published GHCR images. The examples below pin CUDA 13.1.0 on
Ubuntu 24.04; other published tags use the same
`cuda-<cuda-version>-ubuntu<ubuntu-version>` format.

Run the server on the GPU machine:

```bash
docker run --rm --gpus all -p 14833:14833 \
  ghcr.io/lupinemachines/lupine-server:cuda-13.1.0-ubuntu24.04
```

Run the client pointing at that server:

```bash
docker run --rm -it \
  -e LUPINE_SERVER=<server>:14833 \
  ghcr.io/lupinemachines/lupine-client:cuda-13.1.0-ubuntu24.04 \
  nvidia-smi
```

Example output from a real run against a remote RTX 4090:

```text
Mon May 18 15:40:46 2026
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.288.01             Driver Version: 590.48.01    CUDA Version: 13.1     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 4090        On  | 00000000:01:00.0  On |                  Off |
| 30%   52C    P8              22W / 450W |      8MiB / 24564MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
```

Inside the client container, `LD_LIBRARY_PATH=/opt/lupine/lib` is already set,
so CUDA driver users pick up the LUPINE `libcuda.so.1` shim and NVML users such
as `nvidia-smi` pick up the LUPINE `libnvidia-ml.so.1` shim automatically.

## Trace Logging

Set `LUPINE_TRACE` on the client, server, or both to enable trace logging.
`LUPINE_TRACE=0` or an unset value disables tracing. `LUPINE_TRACE=1` writes
trace output to stdout, `LUPINE_TRACE=2` writes it to stderr, and any other
non-empty value is treated as a file path opened in append mode.

```bash
# trace to stdout
LUPINE_TRACE=1 ./your_cuda_program

# trace to stderr
LUPINE_TRACE=2 ./server

# trace to a file
LUPINE_TRACE=/tmp/lupine.trace ./your_cuda_program
```

The same `LUPINE_TRACE` variable controls both client and server tracing;
`LUPINE_SERVER_TRACE` is no longer used.

## Multi-GPU Across Multiple Servers

The client accepts a comma-separated `LUPINE_SERVER` list. Devices are exposed as
one local ordinal list in server order: all GPUs from the first server, then all
GPUs from the next server, and so on.

Run a server on each GPU machine:

```bash
# on gpu-host-a
docker run --rm --gpus all -p 14833:14833 \
  ghcr.io/lupinemachines/lupine-server:cuda-13.1.0-ubuntu24.04

# on gpu-host-b
docker run --rm --gpus all -p 14833:14833 \
  ghcr.io/lupinemachines/lupine-server:cuda-13.1.0-ubuntu24.04
```

Point the client at both servers:

```bash
docker run --rm --network host \
  -e LUPINE_SERVER=gpu-host-a:14833,gpu-host-b:14833 \
  ghcr.io/lupinemachines/lupine-client:cuda-13.1.0-ubuntu24.04 \
  nvidia-smi -L
```

Expected output lists both remote GPUs:

```text
GPU 0: NVIDIA GeForce RTX 4090 (UUID: GPU-...)
GPU 1: NVIDIA GeForce RTX 4090 (UUID: GPU-...)
```

CUDA driver applications use the same `LUPINE_SERVER` value:

```bash
docker run --rm --network host \
  -e LUPINE_SERVER=gpu-host-a:14833,gpu-host-b:14833 \
  ghcr.io/lupinemachines/lupine-client:cuda-13.1.0-ubuntu24.04 \
  ./your_cuda_program
```

Cross-server device-to-device and peer (`cuMemcpyDtoD` / `cuMemcpyPeer`) copies are
supported: when the source and destination live on different servers, the client
transparently stages the data through itself (device->host on one server, then
host->device on the other). Direct server-to-server transfers that avoid that
client hop, cross-server peer-access enablement, and `cuMemcpy3DPeer` are not
implemented yet.
Same-server operations route by handle ownership.

For a specific CUDA version:

```bash
docker pull ghcr.io/lupinemachines/lupine-client:cuda-12.4.1-ubuntu22.04
docker pull ghcr.io/lupinemachines/lupine-server:cuda-12.4.1-ubuntu22.04
```

Client images are also published with a `-slim` tag, for example
`ghcr.io/lupinemachines/lupine-client:cuda-13.1.0-ubuntu24.04-slim`. The
default client tag keeps the CUDA runtime libraries for applications that link
against them; the slim tag includes only the LUPINE shims, their runtime
dependencies, and `nvidia-smi`.

## Slow Start for the Skeptics

This path derives a small PyTorch client image from the published LUPINE client
image and runs the `microgpt_train` test against a remote GPU. It is
intentionally explicit so it is easy to see which side is the CPU-only client
and which side owns the GPU.

Create a PyTorch client Dockerfile in the repo root:

```dockerfile
# Dockerfile.pytorch-lupine
FROM ghcr.io/lupinemachines/lupine-client:cuda-13.1.0-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --break-system-packages \
    --index-url https://download.pytorch.org/whl/cu130 \
    torch

COPY test/pytorch_lupine_tests.py /opt/lupine/test/pytorch_lupine_tests.py

ENV LD_LIBRARY_PATH=/opt/lupine/lib:${LD_LIBRARY_PATH}

CMD ["python3", "/opt/lupine/test/pytorch_lupine_tests.py", "microgpt_train"]
```

Build it:

```bash
docker build -f Dockerfile.pytorch-lupine -t lupine-pytorch:cuda-13.1 .
```

Run the server on the GPU machine:

```bash
docker run --rm --gpus all -p 14833:14833 \
  ghcr.io/lupinemachines/lupine-server:cuda-13.1.0-ubuntu24.04
```

Run the PyTorch client from the CPU-only machine:

```bash
docker run --rm \
  -e LUPINE_SERVER=<server>:14833 \
  lupine-pytorch:cuda-13.1
```

Expected success looks like:

```text
microgpt first_loss=... last_loss=...
microgpt_train: PASS
```

## Local development

Building the binaries requires running codegen first. Lupine codegen reads the cuda dependency header files in order to generate rpc calls.

To ensure codegen works properly, the proper cuda packages need to be installed on your OS. Take a look at our [Dockerfile](./Dockerfile) to see an example.

Take a look [here to install CUDA Toolkit](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_network) (choose your system)

Codegen requires [cuBLAS](https://developer.nvidia.com/hpc-sdk-downloads), [cuDNN](https://developer.nvidia.com/cudnn-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04&target_type=deb_network), [NVML](https://developer.nvidia.com/management-library-nvml), etc:

```py
cudnn_graph_header = find_header_file("cudnn_graph.h")
cudnn_ops_header   = find_header_file("cudnn_ops.h")
cuda_header        = find_header_file("cuda.h")
cublas_header      = find_header_file("cublas_api.h")
cudart_header      = find_header_file("cuda_runtime_api.h")
annotations_header = find_header_file("annotations.h")
```

### Run codegen

```bash
cd codegen && python3 ./codegen.py
```

Ensure there are no errors in the output of the codegen.

### Run cmake

```sh
cmake -S . -B build
cmake --build build
```

CMake builds the CUDA driver shim at `build/libcuda.so.1`, the NVML shim at
`build/libnvidia-ml.so.1`, and the server at `build/lupine_driver_server`.

The Lupine server must be running before initiating client commands.

```sh
./local.sh server
```

If successful, the server will start:

```bash
Server listening on port 14833...
```

## Running the client

For local development, preload the built `libcuda.so.1` before executing CUDA
commands. The published client image sets `LD_LIBRARY_PATH` for you instead.

Once the server above is running:

```sh
# update to your desired IP/port
export LUPINE_SERVER=<server>:14833

LD_PRELOAD=./build/libcuda.so.1 python3 -c "import torch; print(torch.cuda.is_available())"

# or

LD_PRELOAD=./build/libcuda.so.1 nvidia-smi
```

You can also use the local shell script to run your commands.

```
./local.sh run
```

## Questions

1. **What does LUPINE stand for?** Nothing, it just looks cool in all caps.
2. **Does this support authentication? TLS?** Indirectly, yes. It's a plain HTTP/2 server, so you can front it with whatever TLS/auth server you want.
3. **Was this repo AI-generated?** A chunk of it, yes. I mean, would you want to hand write hundreds of tedious API stubs? No? Me neither.
4. **Doesn't this incur a lot of latency?** Surprisingly, no! You will see device transfers get slower because this is basically bottlenecking a PCIe link over the network, but there is very little overhead besides that. For things like model training and inference, once the model is on the GPU very little data transfer happens to the host. As a result, it might be faster than you expect.
5. **Can I do remote video encoding/decoding?** This is probably one use case we wouldn't recommend because that's a lot heavier on the PCIe link. It works in theory though, so if you do have access to a 1 Tbps link it might work for you.

## Prior Art

This project is inspired by some existing proprietary solutions:

- https://www.thundercompute.com/
- https://www.juicelabs.co/
- https://en.wikipedia.org/wiki/RCUDA
