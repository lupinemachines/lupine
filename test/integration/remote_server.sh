#!/usr/bin/env bash
# Shared lifecycle for Linux GPU integration servers. The caller supplies
# SSH_ARGS, SSH_COMMAND_TIMEOUT, SERVER_SSH_TARGET, SERVER_REMOTE_BIN, and
# SERVER_LD_LIBRARY_PATH. The remote host needs iproute2 (ss).

ssh_with_timeout() {
  timeout --kill-after=5s "$SSH_COMMAND_TIMEOUT" \
    ssh "${SSH_ARGS[@]}" "$SERVER_SSH_TARGET" "$@"
}

stop_remote_server() {
  local pidfile="$1"
  local server_log="$2"

  ssh_with_timeout "
    if [ -f '$pidfile' ]; then
      pid=\$(cat '$pidfile' 2>/dev/null || true)
      if [ -n \"\$pid\" ]; then
        kill \"\$pid\" >/dev/null 2>&1 || true
        # pidwait wakes on process exit. Keep polling as a fallback for hosts
        # without pidwait or kernel support, and retain the one-second grace.
        wait_status=0
        timeout 1s pidwait -F '$pidfile' >/dev/null 2>&1 || wait_status=\$?
        if [ \"\$wait_status\" != 0 ] && [ \"\$wait_status\" != 124 ]; then
          for _ in 1 2 3 4 5 6 7 8 9 10; do
            kill -0 \"\$pid\" >/dev/null 2>&1 || break
            sleep 0.1
          done
        fi
        # The listener waits for its connection children, so it is still alive
        # here only when one of them is wedged in a CUDA call it will not
        # return from. List the children before killing the listener: once it
        # is gone they are reparented and unreachable, and a survivor holds a
        # CUDA context, and whatever kernel wedged it, for the rest of the run.
        children=\$(pgrep -P \"\$pid\" 2>/dev/null || true)
        kill -9 \"\$pid\" \$children >/dev/null 2>&1 || true
      fi
    fi
    rm -f '$pidfile' '$server_log'
  " >/dev/null 2>&1 || true
}

start_remote_server() {
  local pidfile="$1"
  local server_log="$2"
  local port="$3"
  local extra_environment="${4:-}"
  local attempt
  local server_environment="LUPINE_PORT=$port"

  if [[ -n "$extra_environment" ]]; then
    server_environment="$extra_environment $server_environment"
  fi

  if [[ -n "$SERVER_LD_LIBRARY_PATH" ]]; then
    printf -v server_environment 'LD_LIBRARY_PATH=%q %s' \
      "$SERVER_LD_LIBRARY_PATH" "$server_environment"
  fi

  for attempt in 1 2 3; do
    stop_remote_server "$pidfile" "$server_log"
    if ssh_with_timeout "
      rm -f '$server_log' '$pidfile'
      $server_environment nohup '$SERVER_REMOTE_BIN' >'$server_log' 2>&1 < /dev/null &
      echo \$! >'$pidfile'
      # Checking the listener avoids making an extra RPC connection.
      for ready_attempt in \$(seq 1 200); do
        pid=\$(cat '$pidfile')
        kill -0 \"\$pid\" 2>/dev/null || exit 1
        if ss -H -ltnp 'sport = :$port' | grep -Fq \"pid=\$pid,\"; then exit 0; fi
        sleep 0.01
      done
      echo 'Server did not listen on port $port' >&2
      exit 1
    "; then
      return 0
    fi
    if (( attempt < 3 )); then
      sleep "$attempt"
    fi
  done

  ssh_with_timeout "cat '$server_log'" >&2 || true
  stop_remote_server "$pidfile" "$server_log"
  return 1
}
