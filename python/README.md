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
LUPINE server. Pass a persistent id when opening the session. The client sends
the id before the HTTP/2 RPC session starts; the server restores the saved
worker process if the artifact exists. When the context exits after CUDA has
initialized, LUPINE saves the worker under the same id and exits that server
worker.

```python
snapshot_id = "0123456789abcdef0123456789abcdef"

with lupine.connect(host="<server>:14833", snapshot_id=snapshot_id) as s:
    device = s.device()
```

`snapshot_type` controls how the snapshot is used. Writing the device snapshot
is the expensive step (it copies GPU memory through the driver), so read-only
reuse avoids paying it repeatedly:

- `"rw"` (default): restore on enter if the snapshot exists, save on exit.
- `"r"`: restore on enter but never save on exit. Reuse a snapshot repeatedly,
  e.g. serving inference from a preloaded model, without re-saving each time.
- `"w"`: start fresh (do not restore) and save on exit. Use this to create or
  refresh a snapshot.

Because the snapshot captures GPU memory at the same device addresses, a Python
variable that owns GPU tensors keeps working across the boundary: restore brings
its memory back where it was, so you don't reload or re-transfer the model.

```python
model = MyModel()  # on the CPU, before any context

# "w": move the model onto the GPU and snapshot that state on exit.
with lupine.connect(host="<server>:14833", snapshot_id=snapshot_id, snapshot_type="w"):
    model = model.to("cuda")           # weights now live in GPU memory

# "r": restore brings those weights back at the same addresses, so the very
# same `model` object is still usable — no reload, no host->device copy.
with lupine.connect(host="<server>:14833", snapshot_id=snapshot_id, snapshot_type="r"):
    output = model(example_input.to("cuda"))
```

The first context must write the snapshot (`"w"` or `"rw"`) and the reuse
contexts must read it (`"r"`). It does **not** work if:

- both contexts are `"r"`: nothing ever saves, so there is no GPU state to
  restore and `model`'s device tensors become invalid.
- the second context is `"w"`: `"w"` starts fresh (no restore) and overwrites
  the snapshot, so `model`'s GPU memory is not brought back.

Snapshots are stored on the server under
`$LUPINE_SNAPSHOT_DIR/objects/<snapshot_id>/`. The current implementation writes
persistent artifacts with NVIDIA CUDA process checkpointing plus CRIU. CRIU is
required on the server only; clients never read or write snapshot artifacts
directly.
