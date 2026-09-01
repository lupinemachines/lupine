# lupine-auto

This companion package is installed by `lupine[auto]`. It activates
LUPINE's server-selected native shims at Python startup. An explicit
`LUPINE_SERVER` from the Lupine CLI wins; otherwise the hook uses the stable
`https://api.lupine.sh` bootstrap endpoint, which redirects a bound
`LUPINE_SESSION` to its regional gateway.

Set `LUPINE_AUTO=0` to disable the startup hook for a process.
