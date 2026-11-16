import re, sys, statistics, collections

runs = collections.defaultdict(list)
ctrl = {}
fails = 0
for line in open(sys.argv[1]):
    if "E2E dev=" not in line:
        continue
    d = dict(re.findall(r'(\w+)=("[^"]*"|\S+)', line))
    data = d["file"].split("/")[-1].removesuffix(".bin")
    key = (data, d["gpu"], d["cfg"])
    fails += d["verify"] != "ok"
    if d["reps"] == "0":
        ctrl[key] = int(d["server_ticks"])
        continue
    gib = int(d["reps"]) * 0.5
    runs[key].append((float(d["gibs_med"]), int(d["server_ticks"]), float(d.get("client_cpu_s_per_gib", "nan")), gib))

cfgs = ["off", "zstd64-cpu", "zstd64", "zstd256", "lz4-64"]
names = {"off": "CPU decode, stream (today)", "zstd64-cpu": "CPU decode, zstd 64K frames",
         "zstd64": "GPU zstd 64K", "zstd256": "GPU zstd 256K", "lz4-64": "GPU LZ4 64K"}
datasets = ["zeros", "tokens_i64", "text", "f32_sparse90", "bf16_weights", "random"]
print("| data | GPU | layout / decode | GiB/s median (min-max) | vs today | server CPU s/GiB | client CPU s/GiB |")
print("|---|---|---|---:|---:|---:|---:|")
for ds in datasets:
    for g in ("2070", "4090"):
        base = None
        for c in cfgs:
            v = runs.get((ds, g, c))
            if not v:
                continue
            rate = [x[0] for x in v]
            m = statistics.median(rate)
            base = m if c == "off" else base
            c0 = ctrl.get((ds, g, c), 0)
            srv = statistics.median([(t - c0) / 100 / gib for _, t, _, gib in v])
            cli = statistics.median([x[2] for x in v])
            rel = f"{m / base:.2f}x" if base else "-"
            print(f"| {ds} | {g} | {names[c]} | {m:.2f} ({min(rate):.2f}-{max(rate):.2f}, n={len(v)}) | {rel} | {max(srv, 0):.3f} | {cli:.3f} |")
print(f"\nverify failures: {fails}")
