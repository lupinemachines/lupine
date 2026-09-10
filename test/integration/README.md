# L4 topology integration tests

This suite provisions two GCP VMs with two NVIDIA L4 GPUs each and one CPU client
VM. Each run selects client/server platforms, topology layouts and workload modes.
`CMakeLists.txt` continues to register the existing single-server sample suites.

## Fleet and layouts

Terraform fixes `a` and `b` to `g2-standard-24` (two L4s each). The CPU `client`
uses `e2-standard-4` for x86_64 or `t2a-standard-4` for arm64. General machine-type
selection and heterogeneous GPU hardware are deferred.
See the [GCP machine specifications](https://docs.cloud.google.com/compute/docs/accelerator-optimized-machines)
for the L4 shape.

`topologies.yaml` describes server processes, GPU assignments, the client's local
GPUs, endpoint order and the exact expected device sequence. The cases cover:

- One remote GPU and two GPUs behind one server.
- Two servers on the same VM or on different VMs.
- Unequal `[2,1]` and equal `[2,2]` server GPU counts.
- Mixed local/remote GPUs on one VM or on different VMs.
- Reversed endpoint order, reversed GPU order and a nonzero physical GPU slot.

Host roles denote distinct VMs. Multiple processes on the same role share that
VM. GPU slots are assigned once by sorting the native GPU UUIDs and are then
resolved to UUIDs for every launch. `a/1` means physical slot 1 on `a`, not ordinal
1 inside a server process. `expect_devices` lists local GPUs first, followed by
each server's GPUs in client endpoint order. Each GPU belongs to only one process
within a case. Cases reuse the fleet sequentially.

## Sparse platform coverage

`runs.yaml` selects the OS/version and CPU architecture independently for each
host, then maps topology names to exact workload modes. It does not expand a
Cartesian product. A GPU host can be a server in one case and a local-GPU client
in another; its platform stays fixed for that run.

| Run | CPU client | GPU host A | GPU host B | Coverage |
| --- | --- | --- | --- | --- |
| `regression` | Ubuntu 24.04 x86_64 | Ubuntu 24.04 x86_64 | Ubuntu 24.04 x86_64 | 172 driver checks across 11 layouts, plus 56 upstream sample invocations including native controls |
| `linux-versions` | Ubuntu 22.04 x86_64 | Ubuntu 24.04 x86_64 | Ubuntu 22.04 x86_64 | Kernel on three selected layouts; 5 checks including native controls |
| `arm64-client` | Ubuntu 24.04 arm64 | Ubuntu 24.04 x86_64 | Ubuntu 22.04 x86_64 | Kernel on three selected remote layouts; 5 checks including native controls |

For example, `remote-two-servers-different-hosts: [kernel]` runs just one workload
on that topology. `kernel` verifies device order, context selection, allocation,
module load, launch and output bytes on every visible GPU. Choosing `[peer-copy]`
runs only synchronous copies, for every ordered device pair. Native controls run
the union of selected workload modes once on each GPU host.

The runner executes up to **eight client processes concurrently** within each
topology, including native controls. `--jobs 1` restores serial execution for
debugging. Each process retains its own timeout, log and result. Only the result
collector writes JSON/JUnit, and every client finishes before servers stop or
the next topology reassigns GPUs. Platform fleets remain sequential to reuse quota.

Linux binaries build on native GitHub x86_64 and arm64 runners. The SSH runner
checks `/etc/os-release` and the actual CPU architecture before selecting the
matching artifact. Requested and observed platforms, selected tests and pinned
images appear in `resolved.json`.

The implemented platform set is Ubuntu 22.04/24.04 x86_64 and an Ubuntu 24.04
arm64 CPU client. [T2A provides Arm CPUs](https://docs.cloud.google.com/compute/docs/general-purpose-machines#t2a_machines);
the current L4 G2 server fleet remains x86_64. Windows/macOS and Arm GPU servers
need their corresponding provision/build/run support before adding executable
cases. Unsupported platform selections fail validation rather than being skipped.

## Tests and P2P failures

`topology_test.cpp` uses the CUDA driver API. It checks exact UUID enumeration,
context/device selection, allocation/copy contents, asynchronous copies ordered
by streams/events, and a deterministic kernel on every device. Each ordered GPU
pair gets separate peer-capability, `cuMemcpyPeer`, and `cuMemcpyPeerAsync` tests.

`run_topologies.py` first runs the workloads against native CUDA on both GPU VMs,
recording each directed peer-access matrix. It then starts fresh servers for each
Lupine layout and runs fresh client processes against them. Within a server,
peer-access reporting must match native CUDA. Across endpoints or a local/remote
boundary, peer enablement must report unsupported, matching the current Lupine
contract. Peer copies are still attempted and their full contents verified.

Every workload process must attest that `cuInit` and `cuGetProcAddress` resolve
to the exact deployed driver library. `driver_guard.so` is preloaded alongside
Lupine and verifies the loaded ELF object's file identity before the test starts;
native controls instead require the NVIDIA driver identified during inventory.
The guard only emits evidence from the intended executable, so an SSH helper or
`timeout` process cannot satisfy the check. Missing evidence fails the test even
if the process exits zero. Remote-only clients also have no local GPU, and exact
UUID checks enforce device routing, including mixed local/remote layouts.

`samples.yaml` pins NVIDIA CUDA Samples 12.4.1 and records build paths, arguments
and timeouts. Select a sample with, for example, `sample:simpleMultiGPU` in
`runs.yaml`. The regression run enables `simpleP2P`, `p2pBandwidthLatencyTest`,
`streamOrderedAllocationP2P`, `simpleMultiGPU`, `MonteCarloMultiGPU`,
`simpleCUFFT_MGPU`, `simpleCUFFT_2d_MGPU` and `conjugateGradientMultiDeviceCG` on
one-server, split-server and mixed local/remote two-GPU layouts. Each has native
controls on both GPU hosts. These run the unmodified upstream CUDA runtime
samples over Lupine's driver shim; they do not test the separate CUDART proxy.

Samples get a UUID preflight under the same environment and the driver guard
runs inside the actual sample executable. The two direct-peer samples may exit
with upstream `EXIT_WAIVED` when no peer-capable pair exists. This is recorded as
a JUnit skip, never a pass, and is accepted only when the native peer matrix and
topology ownership confirm the missing capability. An unexpected waiver fails.
The explicit driver peer-copy tests still exercise copies across those boundaries.

P2P copies are not skipped when native peer access is unavailable. A failed copy,
wrong capability, timeout, segfault or byte mismatch fails its individual test;
remaining tests continue. Native failures are retained too, so reports can show
whether the same operation fails natively or only through Lupine. These failures
make CI red and are not marked as expected successes.

## CI

The `L4 topology integration` job lives in
`.github/workflows/gpu-integration-gcloud.yml`, using its existing GCP identity and
network permissions. It:

1. Validates the sparse run selections and builds Lupine plus the workloads with
   CUDA 12.4.1 on each required native CPU architecture.
2. Resolves CPU/GPU image families to concrete image IDs for the run.
3. Provisions the fleet with Terraform, retrying supported zones after destroying
   any partially allocated fleet.
4. Runs native baselines and selected layouts/workloads with eight client workers,
   collecting results after each test.
5. Destroys the fleet even after test failure and uploads JUnit, per-test logs,
   server logs, native topology diagnostics, build revision/hash and resolved GPU
   mappings/environments.

Platform runs provision one fleet at a time to reuse GPU quota. A failing run
does not cancel other platform runs. Builds are shared across runs.

Terraform uses an existing auto-mode network. SSH is allowed from the CI runner's
IP; RPC ports 14833–14932 are allowed between this run's tagged VMs. A deny rule
blocks other inbound traffic to these VMs. The GCS state prefix is unique per run.
Optional `reservations` can select existing matching capacity for GPU hosts `a`
and `b`; unreserved capacity remains subject to GCP availability and quota.

The startup script installs host runtime dependencies. The selected GPU image
must already have a compatible NVIDIA driver. CI sets a two-hour VM deletion
limit as a fallback. Cleanup tries Terraform first, then deletes this run's VMs
and firewall rules by their unique names if the state remains locked after an
interruption. A stale state lock can require later reconciliation; cleanup does
not force-unlock a possible Terraform writer.

## Local validation and execution

From the repository root, validate without GPUs or cloud credentials:

```bash
python3 -m pip install -r test/integration/requirements.txt
python3 test/integration/validate_specs.py
python3 -m unittest discover -s test/integration -p 'test_*.py'
terraform -chdir=test/integration/terraform/gcp fmt -check
terraform -chdir=test/integration/terraform/gcp init -backend=false
terraform -chdir=test/integration/terraform/gcp validate
```

For a manual run, build the same artifacts CI uses on each required native
architecture (set `topology_arch` to `x86_64` or `arm64` to match that host):

```bash
docker buildx build -f test/integration/Dockerfile \
  --output type=local,dest=/tmp/lupine-topology-artifacts .
mkdir -p /tmp/lupine-topology-builds
tar czf "/tmp/lupine-topology-builds/linux-$topology_arch.tar.gz" -C /tmp/lupine-topology-artifacts .
```

Choose a run from `runs.yaml`. `configure_fleet.py` resolves each host's image
family to a concrete image ID, checks the image architecture and writes Terraform
inputs; its output lists the candidate zones for that run:

```bash
python3 test/integration/configure_fleet.py --run arm64-client \
  --project YOUR_PROJECT --run-id YOUR_RUN_ID \
  --public-key /path/to/public-key --runner-ip YOUR_PUBLIC_IPV4 \
  --output test/integration/terraform/gcp/run.tfvars.json
```

Alternatively, in `terraform/gcp/`, copy `run.tfvars.example` to `run.tfvars` and
fill in the project, unique run ID, per-host platforms/images, public SSH key and
runner IP. Keep platforms consistent with the selected run.
Use an existing GCS bucket and GCP credentials with the required VM/firewall
permissions. Initialize a distinct state prefix for every concurrent fleet:

```bash
terraform init -reconfigure \
  -backend-config="bucket=YOUR_STATE_BUCKET" \
  -backend-config="prefix=gpu-integration/topologies/YOUR_RUN_ID"
terraform plan -var-file=run.tfvars.json -var=zone=YOUR_SELECTED_ZONE -out=run.tfplan
terraform apply run.tfplan
terraform output -json inventory > inventory.json
```

Back at the repository root, execute against that inventory:

```bash
python3 test/integration/run_topologies.py \
  --run arm64-client \
  --inventory test/integration/terraform/gcp/inventory.json \
  --ssh-key /path/to/private-key \
  --artifacts /tmp/lupine-topology-builds \
  --results test/integration/results
```

Finally, from `terraform/gcp/`, run
`terraform destroy -var-file=run.tfvars.json -var=zone=YOUR_SELECTED_ZONE` using
the same state prefix (use `run.tfvars` instead if you filled the example manually).
Keep the state until cleanup succeeds. Results contain
public connection information and GPU mappings; credentials and private SSH keys
stay outside the build and result artifacts.
