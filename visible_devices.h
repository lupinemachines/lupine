#pragma once

#include <functional>
#include <string>
#include <vector>

// Reads CUDA_VISIBLE_DEVICES and answers which devices a client should see and
// in what order, as indices into the full device list.
//
// spec is the raw value of the variable; pass nullptr when it is not set.
// Unset selects everything. An empty value, or "-1", selects nothing. Entries
// are physical indices, or UUIDs with or without a "GPU-" prefix and with or
// without hyphens, matched case-insensitively. Entries that name nothing are
// dropped.
//
// uuid_hex is asked for a device's UUID as lowercase hex with no separators,
// and only if the spec names one, so a list of plain indices costs no lookups.
// It may return an empty string for a device whose UUID cannot be read.
std::vector<int> lupine_select_visible_devices(
    const char *spec, int count,
    const std::function<std::string(int)> &uuid_hex);
