#ifndef LUPINE_PROCESS_HANDOFF_H
#define LUPINE_PROCESS_HANDOFF_H

#include <cstdint>
#include <string>

struct conn_t;

// Private lifecycle protocol shared by the connection worker and transport.
constexpr int LUPINE_HANDOFF_DRAIN_REQUEST = 0x4c504d02;
constexpr uint8_t LUPINE_HANDOFF_CONTROL_FRAME = 0xf0;

// Bracket every request builder. While requests are paused, begin waits,
// except for callbacks and the drain request; a pause returns once no builder
// is open. Replies not yet read do not hold it up.
int lupine_handoff_request_begin(conn_t *conn, int op, bool callback);
void lupine_handoff_request_end(conn_t *conn);
int lupine_handoff_pause_requests(conn_t *conn);
void lupine_handoff_resume_requests(conn_t *conn);
void lupine_handoff_on_resume(conn_t *conn, void (*callback)());

int lupine_handoff_send_control(conn_t *conn, char command,
                                const std::string &checkpoint);
int lupine_handoff_receive_control(conn_t *conn, char *command,
                                   std::string *checkpoint);
int lupine_handoff_park_socket(conn_t *conn);
int lupine_handoff_resume_socket(conn_t *conn);

#endif
