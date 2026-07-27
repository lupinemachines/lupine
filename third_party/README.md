# Third-party dependencies

## LZ4

[LZ4](https://github.com/lz4/lz4) is vendored in `third_party/lz4` as a
squashed git subtree. The currently imported release is v1.10.0.

To update it from the repository root, replace the tag below with the desired
release:

```sh
git subtree pull \
  --prefix=third_party/lz4 \
  https://github.com/lz4/lz4.git \
  v1.10.0 \
  --squash
```

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
