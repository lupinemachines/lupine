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

[Zstandard](https://github.com/facebook/zstd) v1.5.7 is pinned to upstream commit
`f8745da6ff1ad1e7bab384bd1f9d742439278e99`. We vendor the unchanged `lib/common`,
`lib/compress`, and `lib/decompress` directories, the two public codec headers,
and both upstream license files. Upstream programs, tests, benchmarks, and
contributed tools are not part of Lupine's build and are omitted.

The neutral transport builds these C sources statically without compression
worker threads or legacy codecs. The client also uses the bundled xxHash for
profile-cache keys. No system Zstd dependency or user configuration is needed.

To update, replace the tag below with the desired release, verify its commit,
and reimport the same subset. This replaces the earlier full-subtree workflow
so updates do not reintroduce unused upstream tooling:

```sh
git fetch https://github.com/facebook/zstd.git v1.5.7
git rev-parse FETCH_HEAD
git rm -r third_party/zstd
mkdir -p third_party/zstd
git archive FETCH_HEAD LICENSE COPYING \
  lib/zstd.h lib/zstd_errors.h lib/common lib/compress lib/decompress \
  | tar -x -C third_party/zstd
git add third_party/zstd
```

Update the version and commit recorded here and the version in `NOTICE`, then
rebuild and run the transport tests, including ASan/UBSan and GPU integration.
