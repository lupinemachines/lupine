#!/usr/bin/env bash
# Temporary diagnostics for the intermittent nvCOMP quickstart CI timeout.
# Remove this file and its remote_server.sh hooks after the cause is identified.
lupine_sample_watchdog=""

lupine_stop_sample_diagnostics() {
  if [[ -n "$lupine_sample_watchdog" ]]; then
    kill "$lupine_sample_watchdog" 2>/dev/null || true
    wait "$lupine_sample_watchdog" 2>/dev/null || true
    lupine_sample_watchdog=""
  fi
}

lupine_start_sample_diagnostics() {
  [[ "${unit:-}" == nvCOMP/examples/dynamic_bin/high_level_quickstart_example &&
     -n "${RESULTS_DIR:-}" ]] || return 0
  local pidfile="$1" server_log="$2"
  local delay=$(( ${timeout_seconds:-120} - 35 ))
  (( delay > 0 )) || delay=1
  (
    timer=""
    trap 'if [[ -n "$timer" ]]; then kill "$timer" 2>/dev/null || true; wait "$timer" 2>/dev/null || true; fi; exit 0' TERM INT
    # Snapshot identity before the delay, so port/PID reuse cannot attach the
    # debugger to a later test. Cancelling the watchdog also cancels its timer.
    identity=$(SSH_COMMAND_TIMEOUT=5 ssh_with_timeout "
      pid=\$(cat '$pidfile' 2>/dev/null) || exit 1
      printf '%s ' \"\$pid\"
      awk '{print \$22}' /proc/\$pid/stat
    ") || exit 0
    [[ "$identity" =~ ^[0-9]+[[:space:]][0-9]+$ ]] || exit 0
    read -r listener started <<<"$identity"
    sleep "$delay" &
    timer=$!
    wait "$timer" || exit 0
    timer=""
    # Bound the entire SSH call (including all attaches) below the remaining
    # test deadline. Children carry the CUDA work; inspect them first.
    SSH_COMMAND_TIMEOUT=20 ssh_with_timeout "
      [ \"\$(cat '$pidfile' 2>/dev/null)\" = '$listener' ] || exit 0
      [ \"\$(awk '{print \$22}' /proc/$listener/stat 2>/dev/null)\" = '$started' ] || exit 0
      executable=\$(readlink -f '$SERVER_REMOTE_BIN')
      [ \"\$(readlink /proc/$listener/exe)\" = \"\$executable\" ] || exit 0
      count=0
      for target in \$(pgrep -P '$listener') '$listener'; do
        [ \"\$(readlink /proc/\$target/exe)\" = \"\$executable\" ] || continue
        if [ \"\$target\" != '$listener' ]; then
          [ \"\$(awk '/^PPid:/ {print \$2}' /proc/\$target/status)\" = '$listener' ] || continue
        fi
        ps -L -p \"\$target\" -o pid,tid,stat,wchan:32,comm
        for channel in /proc/\$target/task/*/wchan; do
          printf '%s: ' \"\$channel\"
          cat \"\$channel\" 2>/dev/null || true
          printf '\n'
        done
        if command -v gdb >/dev/null 2>&1; then
          debugger=gdb
          if [ \"\$(id -u)\" != 0 ] && [ \"\$(cat /proc/sys/kernel/yama/ptrace_scope 2>/dev/null || echo 0)\" != 0 ]; then
            debugger='sudo -n gdb'
          fi
          timeout --kill-after=1s 6s \$debugger -q -nx -batch \
            -iex 'set auto-load off' -ex 'set pagination off' \
            -ex 'thread apply all bt' -ex detach -p \"\$target\" || true
        fi
        count=\$((count + 1))
        [ \"\$count\" -lt 3 ] || break
      done
    " || true
  ) >>"$RESULTS_DIR/${server_log##*/}.stacks" 2>&1 &
  lupine_sample_watchdog=$!
}
