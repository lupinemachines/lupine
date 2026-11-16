#!/bin/bash
# lockrun.sh <cmd...>: runs cmd locally while holding node-006's shared bench lock.
d=$(mktemp -d)
mkfifo "$d/in"
exec 7<>"$d/in"
ssh -T kevin@inferable-node-006 'flock /tmp/lupine-agent-bench.lock sh -c "echo LOCKED; read x"' <&7 >"$d/out" &
pid=$!
until grep -q LOCKED "$d/out" 2>/dev/null; do sleep 0.2; done
"$@"
rc=$?
echo done >&7
wait $pid
exec 7>&-
rm -rf "$d"
exit $rc
