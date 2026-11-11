#pragma once

#include <cstdint>
#include <string>

struct conn_t;

struct lupine_captured_stdout {
  bool active = false;
  uint64_t wire_size = 0;
  std::string output;
};

void lupine_require_stdout_capture();
bool lupine_start_stdout_capture(lupine_captured_stdout *capture,
                                 bool force = false);
void lupine_finish_stdout_capture(lupine_captured_stdout *capture);
int lupine_write_captured_stdout(conn_t *conn,
                                 const lupine_captured_stdout &capture);
int lupine_read_captured_stdout(conn_t *conn);
