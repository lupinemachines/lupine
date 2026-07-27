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
