#ifdef NDEBUG
#undef NDEBUG
#endif
#include "monitoring.h"
#include <cassert>
#include <cstring>
#include <iostream>
int main() {
  assert(lupine_valid_workload_id(nullptr, 0));
  assert(lupine_valid_workload_id("job:worker", 10));
  assert(lupine_valid_workload_id("\xe4\xb8\xad", 3));
  assert(!lupine_valid_workload_id("\xed\xa0\x80", 3));
  assert(!lupine_valid_workload_id("\xf4\x90\x80\x80", 4));
  assert(!lupine_valid_workload_id("\xc0\xaf", 2));
  assert(!lupine_valid_workload_id("\xe4\xb8", 2));
  assert(!lupine_valid_workload_id("a\0b", 3));
  lupine_client_metadata_extended payload = {};
  payload.client.client_pid = 123;
  payload.workload_id_size = 3;
  memcpy(payload.workload_id, "job", 3);
  lupine_client_metadata client = {};
  char id[LUPINE_WORKLOAD_ID_MAX_BYTES + 1];
  lupine_client_metadata_header header = {
      1, static_cast<uint32_t>(
             offsetof(lupine_client_metadata_extended, workload_id) + 3)};
  assert(lupine_decode_client_metadata(header, &payload, &client, id));
  assert(client.client_pid == 123 && strcmp(id, "job") == 0);
  ++header.payload_size;
  assert(lupine_decode_client_metadata(header, &payload, &client, id) &&
         id[0] == 0);
  header = {1, sizeof(lupine_client_metadata)};
  assert(lupine_decode_client_metadata(header, &payload, &client, id) &&
         id[0] == 0);
  header = {99, sizeof(payload)};
  assert(!lupine_decode_client_metadata(header, &payload, &client, id));
  header = {1, sizeof(lupine_client_metadata) - 1};
  assert(!lupine_decode_client_metadata(header, &payload, &client, id));
  header = {1, LUPINE_CLIENT_METADATA_MAX_PAYLOAD + 1};
  assert(!lupine_decode_client_metadata(header, &payload, &client, id));
  std::cout << "monitor_metadata_test: PASS\n";
}
