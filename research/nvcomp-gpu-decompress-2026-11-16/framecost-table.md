| data | today: zstd -1 stream | zstd -1 64K | zstd -1 256K | zstd -1 1M | LZ4 64K | LZ4 256K | LZ4 1M |
|---|---:|---:|---:|---:|---:|---:|---:|
| zeros | 32648x, 0.43 s/GiB | 2310x, 0.44 s/GiB | 6991x, 0.38 s/GiB | 15888x, 0.37 s/GiB | 237x, 0.36 s/GiB | 251x, 0.34 s/GiB | 253x, 0.38 s/GiB |
| tokens_i64 | 5.27x, 2.07 s/GiB | 5.13x, wire +3%, 1.49 s/GiB | 5.34x, wire -1%, 1.49 s/GiB | 5.48x, wire -4%, 1.42 s/GiB | 5.63x, wire -6%, 0.71 s/GiB | 5.98x, wire -12%, 0.81 s/GiB | 6.07x, wire -13%, 0.71 s/GiB |
| text | 4.89x, 1.49 s/GiB | 4.01x, wire +22%, 1.45 s/GiB | 4.76x, wire +3%, 1.54 s/GiB | 4.82x, wire +2%, 1.31 s/GiB | 3.31x, wire +48%, 1.01 s/GiB | 3.58x, wire +37%, 1.14 s/GiB | 3.64x, wire +34%, 1.18 s/GiB |
| f32_sparse90 | 7.09x, 1.42 s/GiB | 7.09x, wire +0%, 0.96 s/GiB | 7.12x, wire -0%, 0.98 s/GiB | 7.08x, wire +0%, 1.04 s/GiB | 5.35x, wire +33%, 0.66 s/GiB | 5.38x, wire +32%, 0.75 s/GiB | 5.40x, wire +31%, 0.74 s/GiB |
| bf16_weights | 1.00x, 0.47 s/GiB | 1.00x, wire +0%, 0.63 s/GiB | 1.00x, wire +0%, 0.65 s/GiB | 1.00x, wire +0%, 0.55 s/GiB | 1.00x, wire +0%, 0.44 s/GiB | 1.00x, wire +0%, 0.40 s/GiB | 1.00x, wire +0%, 0.54 s/GiB |
| f32_normal | 1.00x, 0.62 s/GiB | 1.00x, wire +0%, 0.74 s/GiB | 1.00x, wire +0%, 0.70 s/GiB | 1.00x, wire +0%, 0.59 s/GiB | 1.00x, wire +0%, 0.34 s/GiB | 1.00x, wire +0%, 0.19 s/GiB | 1.00x, wire +0%, 0.17 s/GiB |
| random | 1.00x, 0.14 s/GiB | 1.00x, wire +0%, 0.32 s/GiB | 1.00x, wire +0%, 0.48 s/GiB | 1.00x, wire +0%, 0.25 s/GiB | 1.00x, wire +0%, 0.29 s/GiB | 1.00x, wire +0%, 0.30 s/GiB | 1.00x, wire +0%, 0.45 s/GiB |
