import re, sys, statistics, collections

runs = collections.defaultdict(list)
bad = []
for line in open(sys.argv[1]):
    if "E2E dev=" not in line:
        if line.strip() and "E2E_DONE" not in line:
            bad.append(line.strip())
        continue
    d = dict(re.findall(r'(\w+)=("[^"]*"|\S+)', line))
    data = d["file"].split("/")[-1].removesuffix(".bin")
    server = "nvCOMP" if d["port"] in ("16251", "16252") else "CPU-decode"
    cfg = "off" if d["nvcomp"] == "off" else f"framed {d['nvcomp']}K / {server}"
    if d["nvcomp"] == "off" and server == "CPU-decode":
        cfg = "off (2nd server)"
    runs[(data, d["gpu"], cfg)].append((float(d["gibs_med"]), d["verify"]))

datasets = ["zeros", "tokens_i64", "text", "f32_sparse90", "bf16_weights", "f32_normal", "random"]
cfgs = ["off", "framed 64K / nvCOMP", "framed 256K / nvCOMP", "framed 64K / CPU-decode"]
print("| data | GPU | " + " | ".join(cfgs) + " | best/off |")
print("|---|---|" + "---:|" * len(cfgs) + "---:|")
for ds in datasets:
    for g in ("2070", "4090"):
        cells, meds = [], {}
        for c in cfgs:
            v = runs.get((ds, g, c))
            if not v:
                cells.append("-")
                continue
            xs = [x for x, _ in v]
            ok = all(o == "ok" for _, o in v)
            m = statistics.median(xs)
            meds[c] = m
            cells.append(f"{m:.2f} ({min(xs):.2f}-{max(xs):.2f}, n={len(xs)}){'' if ok else ' VERIFY FAIL'}")
        best = max((v for k, v in meds.items() if k != "off"), default=0)
        ratio = f"{best / meds['off']:.2f}x" if "off" in meds else "-"
        print(f"| {ds} | {g} | " + " | ".join(cells) + f" | {ratio} |")
if bad:
    print("\nnon-result lines:", *bad[:10], sep="\n")
