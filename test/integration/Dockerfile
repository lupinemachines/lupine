FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build libnghttp2-dev libssl-dev git python3-yaml ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
RUN cmake -S . -B /build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DLUPINE_BUILD_HIP=OFF \
      -DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
    && cmake --build /build --parallel 4 \
      --target lupine_cuda_client lupine_driver_server \
    && mkdir /out \
    && cp -L /build/libcuda.so.1 /build/lupine_driver_server /out/ \
    && g++ -std=c++17 -O2 -Wall -Wextra -Werror \
      -I/usr/local/cuda/include test/integration/topology_test.cpp \
      -L/usr/local/cuda/lib64/stubs -lcuda -ldl -o /out/topology_test \
    && g++ -std=c++17 -O2 -Wall -Wextra -Werror -shared -fPIC \
      test/integration/driver_guard.cpp -ldl -o /out/driver_guard.so \
    && nvcc -ptx -arch=compute_89 test/integration/topology_kernel.cu \
      -o /out/topology_kernel.ptx
RUN python3 test/integration/build_samples.py
FROM scratch
COPY --from=build /out/ /
