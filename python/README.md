# lupine Python adapter

This package provides small PyTorch helpers for LUPINE. It intentionally returns
ordinary `torch.device("cuda:N")` objects so PyTorch continues to use its normal
CUDA dispatch path while LUPINE handles CUDA driver/NVML calls underneath.

Declare all LUPINE hosts before any PyTorch CUDA work:

```python
import lupine
import torch

with lupine.connect(host="<server>:14833"):
    device = torch.device("cuda", 0)
    model = model.to(device)
```

`connect()` loads the LUPINE `libcuda.so.1` from `../build/libcuda.so.1` when
used from this repository. For an installed package, pass `libcuda=...` or set
`LUPINE_LIBCUDA` if the library lives somewhere else:

```python
import torch

with lupine.connect(host="<server>:14833", libcuda="/opt/lupine/libcuda.so.1"):
    device = torch.device("cuda", 0)
```

For multiple LUPINE servers, pass the full host list in one call. `devices()`
uses the native CUDA topology, so it returns every GPU exposed by every server,
not one device per server. When local GPUs are enabled, they come first; remote
GPUs then follow server order and each server's native device order:

```python
import lupine

with lupine.connect(host=["<server-a>:14833", "<server-b>:14833"]) as s:
    gpus = s.devices()
    model0 = model0.to(gpus[0])
    model1 = model1.to(gpus[1])
```

Do not add a second host after tensors have already been moved to the first one.
LUPINE opens connections from `LUPINE_SERVER` when CUDA first initializes, and
later changes to `LUPINE_SERVER` are not picked up by the current process.

Exiting the context restores the previous `LUPINE_SERVER` value.

The adapter does not create a new PyTorch backend such as
`torch.device("lupine")`. A true custom PyTorch device would require registering
PrivateUse1 kernels and backend support. LUPINE already works best when PyTorch
sees CUDA tensors and the LUPINE library is selected through the dynamic linker.
