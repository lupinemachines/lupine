# lupine Python adapter

This package provides small PyTorch helpers for LUPINE. It intentionally returns
ordinary `torch.device("cuda:N")` objects so PyTorch continues to use its normal
CUDA dispatch path while LUPINE handles CUDA driver/NVML calls underneath.

Declare all LUPINE hosts before any PyTorch CUDA work:

```python
import lupine

with lupine.connect(host="<server>:14833") as s:
    device = s.device()
    model = model.to(device)
```

`connect()` loads the LUPINE `libcuda.so.1` from `../build/libcuda.so.1` when
used from this repository. For an installed package, pass `libcuda=...` or set
`LUPINE_LIBCUDA` if the library lives somewhere else:

```python
with lupine.connect(host="<server>:14833", libcuda="/opt/lupine/libcuda.so.1") as s:
    device = s.device()
```

For multiple LUPINE servers, pass the full host list in one call. The order
defines the CUDA ordinals that PyTorch sees:

```python
import lupine

with lupine.connect(host=["<server-a>:14833", "<server-b>:14833"]) as s:
    gpu0, gpu1 = s.devices()
    model0 = model0.to(gpu0)  # cuda:0
    model1 = model1.to(gpu1)  # cuda:1
```

Do not add a second host after tensors have already been moved to the first one.
LUPINE opens connections from `LUPINE_SERVER` when CUDA first initializes, and
later changes to `LUPINE_SERVER` are not picked up by the current process.

Exiting the context restores `LUPINE_SERVER` only if CUDA was not initialized
inside the block. If CUDA was initialized, the process-global LUPINE connection
is already active and cannot be disconnected safely.

The adapter does not create a new PyTorch backend such as
`torch.device("lupine")`. A true custom PyTorch device would require registering
PrivateUse1 kernels and backend support. LUPINE already works best when PyTorch
sees CUDA tensors and the LUPINE library is selected through the dynamic linker.

## Persistent GPU snapshots

Server-side snapshots are enabled by setting `LUPINE_SNAPSHOT_DIR` on the
LUPINE server. If it is unset or empty, snapshot calls return an unsupported
CUDA error.

```python
import lupine

with lupine.connect(host="<server>:14833") as s:
    snapshot_id = s.snapshot()
    status = s.snapshot_status(snapshot_id)
```

For reconnect-boundary restore, pass the persistent id when opening the
session. The client sends the id before the HTTP/2 RPC session starts; the
server restores the saved worker process if the artifact exists. When the
context exits after CUDA has initialized, LUPINE saves the worker under the same
id and exits that server worker.

```python
snapshot_id = "0123456789abcdef0123456789abcdef"

with lupine.connect(host="<server>:14833", snapshot_id=snapshot_id) as s:
    device = s.device()
```

Snapshots are stored on the server under
`$LUPINE_SNAPSHOT_DIR/objects/<snapshot_id>/`. Creation waits until the artifact
has been published and is ready. The current implementation writes persistent
artifacts with NVIDIA CUDA process checkpointing plus CRIU. CRIU is required on
the server only; clients never read or write snapshot artifacts directly.
