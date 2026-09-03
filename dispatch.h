#ifndef LUPINE_DISPATCH_H
#define LUPINE_DISPATCH_H

#include <cstddef>
#include <string>

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
// as a single HTTP/1.x request and returns 1 with the connection fully served.
// HTTP/1.x reads and writes use a short deadline. Returns -1 when the peer
// disconnects before the protocol is decidable or the socket fails; the caller
// closes the socket for every nonzero result.
using lupine_metrics_handler = std::string (*)();

int lupine_connection_dispatch(lupine_socket_t connfd,
                               const rpc_http2_server_metadata *metadata,
                               lupine_metrics_handler metrics = nullptr);

#endif
