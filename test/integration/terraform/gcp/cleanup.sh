#!/usr/bin/env bash
# Release this run's resources even if cancellation left Terraform's state locked.
set -euo pipefail
tf_dir="${1:?Terraform directory required}"
if terraform -chdir="$tf_dir" destroy -auto-approve -lock-timeout=30s \
    -var-file=run.tfvars.json; then
  exit 0
fi

read -r topology_project topology_run_id < <(python3 - "$tf_dir/run.tfvars.json" <<'PY'
import json, sys
config = json.load(open(sys.argv[1]))
print(config['project_id'], config['run_id'])
PY
)
test -n "$topology_project"
test -n "$topology_run_id"

# The state is unique to this run. Leave any stale lock for later reconciliation;
# deleting the owned cloud resources does not require unlocking a possible writer.
topology_instances="$(gcloud compute instances list --project="$topology_project" \
  --filter="name~'^${topology_run_id}-(a|b|client)$'" \
  --format='value(name,zone.basename())')"
if [[ -n "$topology_instances" ]]; then
  while read -r topology_name topology_zone; do
    gcloud compute instances delete "$topology_name" \
      --project="$topology_project" --zone="$topology_zone" --quiet
  done <<< "$topology_instances"
fi

for suffix in ssh rpc deny; do
  topology_rule="${topology_run_id}-${suffix}"
  if gcloud compute firewall-rules describe "$topology_rule" \
      --project="$topology_project" > /dev/null 2> "$tf_dir/cleanup-error.log"; then
    gcloud compute firewall-rules delete "$topology_rule" --project="$topology_project" --quiet
  elif ! grep -q 'was not found' "$tf_dir/cleanup-error.log"; then
    cat "$tf_dir/cleanup-error.log" >&2
    exit 1
  fi
done
