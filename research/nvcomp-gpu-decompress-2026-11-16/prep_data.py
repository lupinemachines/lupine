import json, os, re, struct, sys
import numpy as np

out = sys.argv[1]
st_path = sys.argv[2]
N = 512 << 20
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(1234)


def put(name, b):
    b = bytes(b)
    assert len(b) >= N, (name, len(b))
    with open(os.path.join(out, name + ".bin"), "wb") as f:
        f.write(b[:N])


corpus = bytearray()
for root in ("/usr/include", "/usr/share/doc"):
    for d, _, files in sorted(os.walk(root)):
        for fn in sorted(files):
            p = os.path.join(d, fn)
            if fn.endswith((".gz", ".xz", ".bz2", ".png", ".jpg", ".pdf")) or os.path.islink(p):
                continue
            try:
                with open(p, "rb") as f:
                    data = f.read()
            except OSError:
                continue
            if b"\0" in data[:4096]:
                continue
            corpus += data
print("corpus bytes", len(corpus))
put("text", (corpus * (N // len(corpus) + 1)))

words = re.findall(rb"\w+|[^\w\s]", bytes(corpus))
vocab = {}
for w in words:
    vocab[w] = vocab.get(w, 0) + 1
rank = {w: i for i, (w, _) in enumerate(sorted(vocab.items(), key=lambda kv: -kv[1])[:50000])}
ids = np.fromiter((rank.get(w, 50000) for w in words), dtype=np.int64, count=len(words))
print("tokens", len(ids), "vocab", len(vocab))
put("tokens_i64", np.tile(ids, N // ids.nbytes + 1).tobytes())

with open(st_path, "rb") as f:
    hlen = struct.unpack("<Q", f.read(8))[0]
    f.seek(8 + hlen)
    put("bf16_weights", f.read(N))

put("f32_normal", rng.normal(0, 0.02, N // 4).astype(np.float32).tobytes())
sp = rng.normal(0, 1, N // 4).astype(np.float32)
sp[rng.random(N // 4) < 0.9] = 0
put("f32_sparse90", sp.tobytes())
put("random", rng.integers(0, 256, N, dtype=np.uint8).tobytes())
