| GPU | data | decode | upload alone GiB/s | upload under load GiB/s | upload slowdown | workload kernels/s alone | during upload | workload slowdown |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 2070 | zeros | CPU (today) | 7.91 | 8.07 | 1.0x | 203 | 198 | 3% |
| 2070 | zeros | GPU zstd 64K | 37.38 | 2.28 | 16.4x | 204 | 176 | 13% |
| 2070 | zeros | GPU LZ4 64K | 40.34 | 3.36 | 12.0x | 250 | 221 | 12% |
| 2070 | f32_sparse90 | CPU (today) | 1.59 | 1.59 | 1.0x | 250 | 247 | 1% |
| 2070 | f32_sparse90 | GPU zstd 64K | 1.89 | 1.09 | 1.7x | 200 | 117 | 41% |
| 2070 | f32_sparse90 | GPU LZ4 64K | 1.45 | 1.15 | 1.3x | 204 | 124 | 39% |
| 4090 | zeros | CPU (today) | 2.63 | 2.64 | 1.0x | 187 | 187 | 0% |
| 4090 | zeros | GPU zstd 64K | 38.30 | 2.53 | 15.1x | 188 | 167 | 11% |
| 4090 | zeros | GPU LZ4 64K | 40.88 | 2.93 | 13.9x | 186 | 167 | 10% |
| 4090 | f32_sparse90 | CPU (today) | 1.58 | 1.58 | 1.0x | 188 | 186 | 1% |
| 4090 | f32_sparse90 | GPU zstd 64K | 1.90 | 1.21 | 1.6x | 187 | 119 | 36% |
| 4090 | f32_sparse90 | GPU LZ4 64K | 1.45 | 1.24 | 1.2x | 186 | 123 | 34% |
