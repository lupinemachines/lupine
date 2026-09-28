# Third-party dependencies

## libcuckoo

[libcuckoo](https://github.com/efficient/libcuckoo) is vendored in
`third_party/libcuckoo` as a squashed git subtree. The currently imported
revision is `0b0ffe0718c7995ca2a20266b1c02dd5a0138fde`.

To update it from the repository root, replace the revision below with the
desired commit or tag:

```sh
git subtree pull \
  --prefix=third_party/libcuckoo \
  https://github.com/efficient/libcuckoo.git \
  0b0ffe0718c7995ca2a20266b1c02dd5a0138fde \
  --squash
```

## Zstandard

[Zstandard](https://github.com/facebook/zstd) v1.5.7 is vendored as a squashed
subtree in `third_party/zstd`. The neutral transport builds only its common,
compression and decompression C sources, without worker threads or legacy codecs.
The client also uses its bundled xxHash for profile-cache keys.
No system Zstd dependency or user configuration is needed on any platform.

```sh
git subtree pull --prefix=third_party/zstd https://github.com/facebook/zstd.git v1.5.7 --squash
```
