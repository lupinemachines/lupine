# L4 topology integration tests

This suite provisions two GCP VMs with two NVIDIA L4 GPUs each and one CPU client
VM, then runs the same CUDA driver workloads through eleven Lupine layouts.
`CMakeLists.txt` continues to register the existing single-server sample suites.

## Fleet and layouts

Terraform fixes `a` and `b` to `g2-standard-24` (two L4s each) and `client` to
`e2-standard-4`. Machine-type selection and heterogeneous hardware are deferred.
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

P2P copies are not skipped when native peer access is unavailable. A failed copy,
wrong capability, timeout, segfault or byte mismatch fails its individual test;
remaining tests continue. Native failures are retained too, so reports can show
whether the same operation fails natively or only through Lupine. These failures
make CI red and are not marked as expected successes.

## CI

The `L4 topology integration` job lives in
`.github/workflows/gpu-integration-gcloud.yml`, using its existing GCP identity and
network permissions. It:

1. Validates the matrix and builds Lupine plus the workloads with CUDA 12.4.1.
2. Resolves CPU/GPU image families to concrete image IDs for the run.
3. Provisions the fleet with Terraform, retrying supported zones after destroying
   any partially allocated fleet.
4. Runs native baselines and all eleven layouts, collecting results after each test.
5. Destroys the fleet even after test failure and uploads JUnit, per-test logs,
   server logs, native topology diagnostics, build revision/hash and resolved GPU
   mappings/environments.

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
terraform -chdir=test/integration/terraform/gcp fmt -check
terraform -chdir=test/integration/terraform/gcp init -backend=false
terraform -chdir=test/integration/terraform/gcp validate
```

For a manual run, build the same artifacts CI uses:

```bash
docker buildx build -f test/integration/Dockerfile \
  --output type=local,dest=/tmp/lupine-topology-artifacts .
tar czf /tmp/lupine-topology-artifacts.tar.gz -C /tmp/lupine-topology-artifacts .
```

In `terraform/gcp/`, copy `run.tfvars.example` to `run.tfvars` and fill in the
project, unique run ID, zone, concrete image IDs, public SSH key and runner IP.
Use an existing GCS bucket and GCP credentials with the required VM/firewall
permissions. Initialize a distinct state prefix for every concurrent fleet:

```bash
terraform init -reconfigure \
  -backend-config="bucket=YOUR_STATE_BUCKET" \
  -backend-config="prefix=gpu-integration/topologies/YOUR_RUN_ID"
terraform plan -var-file=run.tfvars -out=run.tfplan
terraform apply run.tfplan
terraform output -json inventory > inventory.json
```

Back at the repository root, execute against that inventory:

```bash
python3 test/integration/run_topologies.py \
  --inventory test/integration/terraform/gcp/inventory.json \
  --ssh-key /path/to/private-key \
  --artifacts /tmp/lupine-topology-artifacts.tar.gz \
  --results test/integration/results
```

Finally, from `terraform/gcp/`, run `terraform destroy -var-file=run.tfvars` using
the same state prefix. Keep the state until cleanup succeeds. Results contain
public connection information and GPU mappings; credentials and private SSH keys
stay outside the build and result artifacts.
