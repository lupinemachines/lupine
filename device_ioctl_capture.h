#pragma once
#include <cstdint>

int lupine_capture_rm_alloc(int (*operation)(void *), void *argument, int *fd,
                            uint32_t *client, uint32_t *memory);
