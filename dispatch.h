#ifndef LUPINE_DISPATCH_H
#define LUPINE_DISPATCH_H

#include <cstddef>

#include "lupine_platform.h"
#include "rpc.h"

// Classifies the first bytes of an accepted connection against the fixed
// 24-byte HTTP/2 client connection preface. Returns 1 when `data` begins with
// the complete preface, -1 when it can never become the preface, and 0 when
// more bytes are needed to decide.
int lupine_h2_preface_check(const unsigned char *data, size_t len);

// Decides what protocol an accepted connection speaks before any RPC state
// exists. The socket is only peeked at: an HTTP/2 preface leaves every byte
// in place for the nghttp2 session and returns 0. Anything else is answered
// as a single HTTP/1.x request (HEAD / and GET / mirror the HTTP/2 version
// probe; every other path is a 404) and returns 1 with the connection fully
// served. Returns -1 when the peer disconnects before the protocol is
// decidable or the socket fails; the caller closes the socket for every
// nonzero result. Reads block: a stalled peer holds only its own
// per-connection child until the transport keepalive declares it dead.
int lupine_connection_dispatch(lupine_socket_t connfd,
                               const rpc_http2_server_metadata *metadata);

#endif
