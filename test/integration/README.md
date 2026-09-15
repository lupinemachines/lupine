# GPU integration dependencies

The GCE workflow keeps a dependency image in the existing GCS cache bucket. It
contains the CUDA toolkit, system packages, compiled NVIDIA samples, PyTorch,
CUDA Python, and its matching upstream test checkout. Each job builds a separate
final image that copies the current repository into `/src`; `/src` and Lupine binaries are never restored from the dependency cache.

The key includes the dependency Dockerfile, the two sample/preparation runners,
CUDA base image, sample ref, PyTorch index, GPU architecture, and cache version.
Normal C++ changes reuse the same image. A missing, corrupt, or unloadable cache
falls back to a cold build. Downloads are checked against their SHA-256 metadata
before Docker loads them.

Relevant pushes to `main` populate the cache. Pull requests only consume it.
Manual runs on `main` can also populate a missing entry. Bump
`INTEGRATION_IMAGE_CACHE_VERSION` in the workflow when intentionally refreshing
floating pip versions or base-image tags. The resolved packages stay fixed in a
cached image until its inputs or version change.

Cache publication follows the dependency build, before the test run, so a
failure in project code does not prevent reuse of successfully built
dependencies. Only the dependency image is saved. The workflow reports restore
and build durations; warm runs still pay for downloading and loading the image.

For a local dependency build from the repository root:

```sh
docker build -f test/integration/Dockerfile.dependencies \
  --build-arg BASE_IMAGE=nvidia/cuda:13.3.1-devel-ubuntu24.04 \
  --build-arg CUDA_SAMPLES_REF=v13.3 \
  --build-arg PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu132 \
  --build-arg CUDA_SAMPLES_ARCH=89 \
  -t lupine-integration-deps:local .
```

Outside the image, `CUDA_PYTHON_SETUP_ONLY=1 test/run_cuda_python_tests.sh`
prepares the selected Python environment and test checkout. Then `LIST_TESTS=1`
lists that checkout without network access or a Lupine build.
