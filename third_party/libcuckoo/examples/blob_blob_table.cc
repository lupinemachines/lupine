// This table was created to illustrate usage of the table with untyped blobs.
// See `int_str_table.cc` for more comments regarding the semantics of creating
// implementation files for use with C programs.

extern "C" {
#include "blob_blob_table.h"
}

// The hashtable uses the generic std::hash template wrapper as the hash
// function and std::equal_to as the equality function for key hashing and
// comparison. Since neither of these templates are specialized for our custom
// key_blob type, we must create our own specializations.

#include <cstring>
#include <functional>

namespace std {
template <> struct hash<key_blob> {
  size_t operator()(const key_blob &kb) const { return *(size_t *)kb.blob; }
};

template <> struct equal_to<key_blob> {
  bool operator()(const key_blob &lhs, const key_blob &rhs) const {
    return memcmp(lhs.blob, rhs.blob, sizeof(lhs.blob)) == 0;
  }
};
}

#include <libcuckoo-c/cuckoo_table_template.cc>
