import re, sys, collections

rows = []
for line in open(sys.argv[1]):
    if not line.startswith("RESULT"):
        continue
    d = dict(re.findall(r'(\w+)=("[^"]*"|\S+)', line))
    d["gpu"] = "4090" if "4090" in d["gpu"] else "2070"
    d["data"] = d["file"].split("/")[-1].removesuffix(".bin")
    for k in ("ratio", "gibs_med", "gibs_min", "gibs_max", "cpu_s_per_gib", "comp_mibps_1t", "kernel_only_gibs", "gpu_dec_ms_per_gib"):
        d[k] = float(d[k])
    rows.append(d)

link = {(r["gpu"], r["threads"]): r["gibs_med"] for r in rows if r["codec"] == "none"}
print("Uncompressed pinned-staged HtoD (codec none):", link)
datasets = ["zeros", "tokens_i64", "text", "f32_sparse90", "bf16_weights", "f32_normal", "random"]
print()
print("| data | GPU | thr | codec/frame | ratio | client comp MiB/s (1 core) | CPU path GiB/s | GPU path GiB/s | GPU/CPU | CPU s/GiB cpu->gpu | GPU kernel GiB/s (whole batch) | verify |")
print("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
key = lambda r: (r["data"], r["gpu"], r["threads"], r["codec"], r["level"], r["chunk_kib"])
by = collections.defaultdict(dict)
for r in rows:
    by[key(r)][r["path"]] = r
for ds in datasets:
    for g in ("2070", "4090"):
        for t in ("1", "4"):
            for k in sorted([k for k in by if k[0] == ds and k[1] == g and k[2] == t and k[3] != "none"],
                            key=lambda k: (k[3], int(k[4]), int(k[5]))):
                c, gp = by[k].get("cpu"), by[k].get("gpu")
                if not c or not gp:
                    continue
                name = f"{k[3]}{'' if k[3] in ('lz4','gdeflate') else ' '+k[4]}/{k[5]}K"
                v = "ok" if c["verify"] == gp["verify"] == "ok" else "FAIL"
                print(f"| {ds} | {g} | {t} | {name} | {c['ratio']:.2f} | {c['comp_mibps_1t']:.0f} | {c['gibs_med']:.2f} | {gp['gibs_med']:.2f} | {gp['gibs_med']/c['gibs_med']:.2f}x | {c['cpu_s_per_gib']:.3f}->{gp['cpu_s_per_gib']:.3f} | {gp['kernel_only_gibs']:.1f} | {v} |")
