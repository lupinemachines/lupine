import re, sys, statistics

cases, cur = [], None
for line in open(sys.argv[1]):
    if line.startswith("CASE"):
        cur = dict(re.findall(r"(\w+)=(\S+)", line))
        cur.update(load=[], up=[], loaded=[], alone=None)
        cases.append(cur)
    elif cur is None:
        continue
    elif line.startswith("ALONE"):
        cur["alone"] = float(re.search(r"gibs_med=([0-9.]+)", line).group(1))
    elif line.startswith("LOADED"):
        cur["loaded"].append(float(re.search(r"gibs_med=([0-9.]+)", line).group(1)))
    elif line.startswith("UPSTART"):
        cur["up"].append([float(line.split()[1]), None])
    elif line.startswith("UPEND"):
        cur["up"][-1][1] = float(line.split()[1])
    elif line.startswith("LOAD "):
        t, k = re.findall(r"=([0-9.]+)", line)
        cur["load"].append((float(t), float(k)))

print("| GPU | data | decode | upload alone GiB/s | upload under load GiB/s | upload slowdown | workload kernels/s alone | during upload | workload slowdown |")
print("|---|---|---|---:|---:|---:|---:|---:|---:|")
for c in cases:
    if not c["up"] or not c["load"]:
        continue
    first = c["up"][0][0]
    alone = [k for t, k in c["load"] if t < first - 0.3][2:]
    during = [k for t, k in c["load"] if any(s + 0.3 < t - 0.25 and t <= e for s, e in c["up"] if e)]
    gpu = "4090" if c["port"] == "16251" else "2070"
    dec = {"off": "CPU (today)", "zstd64": "GPU zstd 64K", "lz4-64": "GPU LZ4 64K"}[c["cfg"]]
    lo = statistics.median(c["loaded"]) if c["loaded"] else float("nan")
    a = statistics.median(alone) if alone else float("nan")
    d = statistics.median(during) if during else float("nan")
    print(f"| {gpu} | {c['data']} | {dec} | {c['alone']:.2f} | {lo:.2f} | {c['alone']/lo:.1f}x | {a:.0f} | {d:.0f} | {100*(1-d/a):.0f}% |")
