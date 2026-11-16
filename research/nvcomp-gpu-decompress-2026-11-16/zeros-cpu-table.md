| data | GPU | layout / decode | GiB/s median (min-max) | vs today | server CPU s/GiB | client CPU s/GiB |
|---|---|---|---:|---:|---:|---:|
| zeros | 2070 | CPU decode, stream (today) | 7.99 (7.92-8.00, n=3) | 1.00x | 0.159 | 0.157 |
| zeros | 2070 | GPU zstd 64K | 34.53 (33.60-36.74, n=3) | 4.32x | 0.011 | 0.112 |
| zeros | 2070 | GPU LZ4 64K | 35.81 (35.46-39.43, n=3) | 4.48x | 0.013 | 0.116 |
| zeros | 4090 | CPU decode, stream (today) | 2.62 (2.61-2.63, n=3) | 1.00x | 0.412 | 0.162 |
| zeros | 4090 | GPU zstd 64K | 34.58 (33.91-34.94, n=3) | 13.20x | 0.011 | 0.113 |
| zeros | 4090 | GPU LZ4 64K | 35.25 (33.37-37.88, n=3) | 13.45x | 0.011 | 0.117 |

verify failures: 0
