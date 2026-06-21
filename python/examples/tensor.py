# /// script
# dependencies = [
#   "numpy",
#   "lupine",
#   "torch",
# ]
# ///

import torch
import lupine


def prompt_endpoint() -> str:
    host = input("LUPINE server host: ").strip()
    while not host:
        host = input("LUPINE server host: ").strip()

    port_text = input("LUPINE server port [14833]: ").strip() or "14833"
    port = int(port_text)
    return f"{host}:{port}"


with lupine.connect(host=prompt_endpoint()) as session:
    device = session.device()
    info = getattr(session, "info", None)
    x = torch.arange(8, device=device, dtype=torch.float32)
    y = (x * 2).cpu()
    if info is None:
        props = torch.cuda.get_device_properties(device)
        info = {
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "gpu": props.name,
        }
    print("cuda available:", info["cuda_available"])
    print("device:", device)
    print("count:", info["device_count"])
    print("gpu:", info["gpu"])
    print("result:", y.tolist())
