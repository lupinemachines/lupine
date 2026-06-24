#include "rpc.h"
#include <algorithm>
#include <iostream>
#include <string.h>

void *_rpc_read_id_dispatch(void *p) {
  conn_t *conn = (conn_t *)p;

  if (pthread_mutex_lock(&conn->read_mutex) < 0)
    return NULL;

  while (!conn->closed) {
    while (conn->read_id != 0 && !conn->closed)
      pthread_cond_wait(&conn->read_cond, &conn->read_mutex);
    if (conn->closed)
      break;

    // the read id is zero so it's our turn to read the next int which is the
    // request id of the next request.
    int bytes = rpc_read(conn, &conn->read_id, sizeof(int));
    if (bytes <= 0 || conn->read_id == 0) {
      conn->closed = 1;
      pthread_cond_broadcast(&conn->read_cond);
      break;
    }
    if (pthread_cond_broadcast(&conn->read_cond) < 0)
      break;
  }
  pthread_mutex_unlock(&conn->read_mutex);
  conn->rpc_thread = 0;
  return NULL;
}

// rpc_read_start waits for a response with a specific request id on the
// given connection. this function is used to wait for a response to a
// request that was sent with rpc_write_end.
//
// it is not necessary to call rpc_read_start() if it is the first call in
// the sequence because by convention, the handler owns the read lock on
// entry.
int rpc_dispatch(conn_t *conn, int parity) {
  if (conn->rpc_thread == 0 &&
      pthread_create(&conn->rpc_thread, nullptr, _rpc_read_id_dispatch,
                     (void *)conn) < 0) {
    return -1;
  }

  if (pthread_mutex_lock(&conn->read_mutex) < 0) {
    return -1;
  }

  int op;

  while (!conn->closed && (conn->read_id < 2 || conn->read_id % 2 != parity))
    pthread_cond_wait(&conn->read_cond, &conn->read_mutex);

  if (conn->closed) {
    pthread_mutex_unlock(&conn->read_mutex);
    return -1;
  }

  if (rpc_read(conn, &op, sizeof(int)) < 0) {
    pthread_mutex_unlock(&conn->read_mutex);
    return -1;
  }

  return op;
}

// rpc_read_start waits for a response with a specific request id on the
// given connection. this function is used to wait for a response to a request
// that was sent with rpc_write_end.
//
// it is not necessary to call rpc_read_start() if it is the first call in
// the sequence because by convention, the handler owns the read lock on entry.
int rpc_read_start(conn_t *conn, int write_id) {
  if (pthread_mutex_lock(&conn->read_mutex) < 0)
    return -1;

  // wait for the active read id to be the request id we are waiting for
  while (!conn->closed && conn->read_id != write_id)
    if (pthread_cond_wait(&conn->read_cond, &conn->read_mutex) < 0)
      return -1;

  if (conn->closed) {
    pthread_mutex_unlock(&conn->read_mutex);
    return -1;
  }

  return 0;
}

int rpc_read(conn_t *conn, void *data, size_t size) {
  return rpc_http2_read(conn, data, size);
}

int rpc_drain(conn_t *conn, size_t size) {
  char buffer[64 * 1024];
  size_t offset = 0;
  while (offset < size) {
    size_t chunk = std::min(sizeof(buffer), size - offset);
    if (rpc_read(conn, buffer, chunk) < 0) {
      return -1;
    }
    offset += chunk;
  }
  return 0;
}

// rpc_read_end releases the response lock on the given connection.
int rpc_read_end(conn_t *conn) {
  int read_id = conn->read_id;
  bool completes_local_request =
      read_id >= 2 && (read_id % 2) == conn->local_request_parity;
  conn->read_id = 0;
  if (pthread_cond_broadcast(&conn->read_cond) < 0 ||
      pthread_mutex_unlock(&conn->read_mutex) < 0)
    return -1;
  if (completes_local_request && pthread_mutex_unlock(&conn->call_mutex) < 0)
    return -1;
  return read_id;
}

// rpc_wait_for_response is a convenience function that sends the current
// request and then waits for the corresponding response. this pattern is
// so common that having this function keeps the codegen much cleaner.
int rpc_wait_for_response(conn_t *conn) {
  int write_id = rpc_write_end(conn);
  if (write_id < 0 || rpc_read_start(conn, write_id) < 0) {
    pthread_mutex_unlock(&conn->call_mutex);
    return -1;
  }
  return 0;
}

// rpc_write_start_request starts a new request builder on the given connection
// index with a specific op code.
//
// only one request can be active at a time, so this function will take the
// request lock from the connection.
int rpc_write_start_request(conn_t *conn, const int op) {
  if (conn->closed) {
    return -1;
  }
  if (pthread_mutex_lock(&conn->call_mutex) < 0) {
    return -1;
  }
  if (conn->closed) {
    pthread_mutex_unlock(&conn->call_mutex);
    return -1;
  }
  if (pthread_mutex_lock(&conn->write_mutex) < 0) {
#ifdef VERBOSE
    std::cout << "rpc_write_start failed due to rpc_open() < 0 || "
                 "conns[index].write_mutex lock"
              << std::endl;
#endif
    pthread_mutex_unlock(&conn->call_mutex);
    return -1;
  }

  conn->write_iov_count = 2;               // skip 2 for the header
  conn->request_id = conn->request_id + 2; // leave the last bit the same
  conn->write_id = conn->request_id;
  conn->write_op = op;
  return 0;
}
// rpc_write_start_request starts a new request builder on the given connection
// index with a specific op code.
//
// only one request can be active at a time, so this function will take the
// request lock from the connection.
int rpc_write_start_response(conn_t *conn, const int read_id) {
  if (conn->closed) {
    return -1;
  }
  if (pthread_mutex_lock(&conn->write_mutex) < 0) {
#ifdef VERBOSE
    std::cout << "rpc_write_start failed due to rpc_open() < 0 || "
                 "conns[index].write_mutex lock"
              << std::endl;
#endif
    return -1;
  }

  conn->write_iov_count = 1; // skip 1 for the header
  conn->write_id = read_id;
  conn->write_op = -1;
  return 0;
}

int rpc_write(conn_t *conn, const void *data, const size_t size) {
  conn->write_iov_framed[conn->write_iov_count] = 0;
  conn->write_iov[conn->write_iov_count++] = {(void *)data, size};
  return 0;
}

int rpc_write_kernel_param_values(conn_t *conn, uint32_t count,
                                  const size_t *sizes, void *const *values) {
  if (conn == nullptr ||
      (count != 0 && (sizes == nullptr || values == nullptr))) {
    return -1;
  }
  for (uint32_t i = 0; i < count; ++i) {
    if (values[i] == nullptr || rpc_write(conn, values[i], sizes[i]) < 0) {
      return -1;
    }
  }
  return 0;
}

int rpc_read_kernel_param_values(conn_t *conn, uint32_t count,
                                 const size_t *offsets, const size_t *sizes,
                                 size_t payload_size, void *storage,
                                 size_t storage_size, void **values) {
  if (conn == nullptr ||
      (count != 0 && (offsets == nullptr || sizes == nullptr ||
                      storage == nullptr || values == nullptr))) {
    return -1;
  }

  size_t expected_payload_size = 0;
  for (uint32_t i = 0; i < count; ++i) {
    expected_payload_size += sizes[i];
  }
  if (payload_size != expected_payload_size) {
    return -1;
  }

  auto *bytes = static_cast<unsigned char *>(storage);
  for (uint32_t i = 0; i < count; ++i) {
    if (offsets[i] + sizes[i] > storage_size) {
      return -1;
    }
    unsigned char *dst = bytes + offsets[i];
    if (sizes[i] != 0 && rpc_read(conn, dst, sizes[i]) < 0) {
      return -1;
    }
    values[i] = dst;
  }
  return 0;
}

int rpc_write_cuda_kernel_node_params(conn_t *conn,
                                      const CUDA_KERNEL_NODE_PARAMS *params) {
  static_assert(sizeof(params->func) == sizeof(uint64_t),
                "CUfunction wire size must be 64-bit");
#if CUDA_VERSION >= 12000
  static_assert(sizeof(params->kern) == sizeof(uint64_t),
                "CUkernel wire size must be 64-bit");
  static_assert(sizeof(params->ctx) == sizeof(uint64_t),
                "CUcontext wire size must be 64-bit");
#endif

  if (conn == nullptr || params == nullptr) {
    return -1;
  }
  if (rpc_write(conn, &params->func, sizeof(uint64_t)) < 0 ||
      rpc_write(conn, &params->gridDimX, sizeof(params->gridDimX)) < 0 ||
      rpc_write(conn, &params->gridDimY, sizeof(params->gridDimY)) < 0 ||
      rpc_write(conn, &params->gridDimZ, sizeof(params->gridDimZ)) < 0 ||
      rpc_write(conn, &params->blockDimX, sizeof(params->blockDimX)) < 0 ||
      rpc_write(conn, &params->blockDimY, sizeof(params->blockDimY)) < 0 ||
      rpc_write(conn, &params->blockDimZ, sizeof(params->blockDimZ)) < 0 ||
      rpc_write(conn, &params->sharedMemBytes,
                sizeof(params->sharedMemBytes)) < 0) {
    return -1;
  }
#if CUDA_VERSION >= 12000
  if (rpc_write(conn, &params->kern, sizeof(uint64_t)) < 0 ||
      rpc_write(conn, &params->ctx, sizeof(uint64_t)) < 0) {
    return -1;
  }
#else
  static const uint64_t zero = 0;
  if (rpc_write(conn, &zero, sizeof(zero)) < 0 ||
      rpc_write(conn, &zero, sizeof(zero)) < 0) {
    return -1;
  }
#endif
  return 0;
}

int rpc_read_cuda_kernel_node_params(conn_t *conn,
                                     CUDA_KERNEL_NODE_PARAMS *params) {
  static_assert(sizeof(params->func) == sizeof(uint64_t),
                "CUfunction wire size must be 64-bit");
#if CUDA_VERSION >= 12000
  static_assert(sizeof(params->kern) == sizeof(uint64_t),
                "CUkernel wire size must be 64-bit");
  static_assert(sizeof(params->ctx) == sizeof(uint64_t),
                "CUcontext wire size must be 64-bit");
#endif

  if (conn == nullptr || params == nullptr) {
    return -1;
  }
  *params = {};
  if (rpc_read(conn, &params->func, sizeof(uint64_t)) < 0 ||
      rpc_read(conn, &params->gridDimX, sizeof(params->gridDimX)) < 0 ||
      rpc_read(conn, &params->gridDimY, sizeof(params->gridDimY)) < 0 ||
      rpc_read(conn, &params->gridDimZ, sizeof(params->gridDimZ)) < 0 ||
      rpc_read(conn, &params->blockDimX, sizeof(params->blockDimX)) < 0 ||
      rpc_read(conn, &params->blockDimY, sizeof(params->blockDimY)) < 0 ||
      rpc_read(conn, &params->blockDimZ, sizeof(params->blockDimZ)) < 0 ||
      rpc_read(conn, &params->sharedMemBytes,
               sizeof(params->sharedMemBytes)) < 0) {
    return -1;
  }
#if CUDA_VERSION >= 12000
  if (rpc_read(conn, &params->kern, sizeof(uint64_t)) < 0 ||
      rpc_read(conn, &params->ctx, sizeof(uint64_t)) < 0) {
    return -1;
  }
#else
  uint64_t unused_kern = 0;
  uint64_t unused_ctx = 0;
  if (rpc_read(conn, &unused_kern, sizeof(unused_kern)) < 0 ||
      rpc_read(conn, &unused_ctx, sizeof(unused_ctx)) < 0) {
    return -1;
  }
#endif
  params->kernelParams = nullptr;
  params->extra = nullptr;
  return 0;
}

// rpc_write_framed queues a payload that the transport LZ4-frames lazily,
// one block at a time, as the bytes are streamed to the socket. The caller's
// buffer must stay valid until rpc_write_end() returns, exactly like
// rpc_write(). See compress.cpp for the framing format.
int rpc_write_framed(conn_t *conn, const void *data, const size_t size) {
  conn->write_iov_framed[conn->write_iov_count] = 1;
  conn->write_iov[conn->write_iov_count++] = {(void *)data, size};
  return 0;
}

// rpc_write_end finalizes the current request builder on the given connection
// index and sends the request to the server.
//
// the request lock is released after the request is sent and the function
// returns the request id which can be used to wait for a response.
int rpc_write_end(conn_t *conn) {
  if (conn->closed) {
    pthread_mutex_unlock(&conn->write_mutex);
    return -1;
  }
  conn->write_iov[0] = {&conn->write_id, sizeof(int)};
  conn->write_iov_framed[0] = 0;
  if (conn->write_op != -1) {
    conn->write_iov[1] = {&conn->write_op, sizeof(unsigned int)};
    conn->write_iov_framed[1] = 0;
  }
  int write_id = conn->write_id;
  int iov_count = conn->write_iov_count;
  struct iovec iov[128];
  unsigned char framed[128];
  memcpy(iov, conn->write_iov, sizeof(struct iovec) * iov_count);
  memcpy(framed, conn->write_iov_framed, iov_count);

  int result = rpc_http2_writev(conn, iov, framed, iov_count);
  pthread_mutex_unlock(&conn->write_mutex);
  return result == 0 ? write_id : -1;
}
