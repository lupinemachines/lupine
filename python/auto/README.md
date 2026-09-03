# lupine-auto

This companion package is installed by `lupine[auto]`. It waits until the first
CUDA consumer import, then acquires a cloud lease and activates LUPINE's
server-selected native client. Authenticate once with `uvx lupine login` or
`python -m lupine login`; headless environments can set `LUPINE_API_TOKEN`.

An explicit `LUPINE_SERVER` or existing `LUPINE_SESSION` wins. Without either,
the hook uses `https://api.lupine.sh`, stores the acquired lease in
`LUPINE_SESSION`, and keeps it alive for the process. Missing authentication
produces a login hint without aborting Python startup.

Set `LUPINE_AUTO=0` to disable the startup hook for a process.
