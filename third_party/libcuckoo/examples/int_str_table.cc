// Include the header file with "C" linkage
extern "C" {
#include "int_str_table.h"
}

// Include the implementation template, which uses the constants defined in
// `int_str_table.h`.  The implementation file defines all functions under "C"
// linkage, and should be linked with the corresponding header to generate a
// linkable library. See `CMakeLists.txt` for an example of how this is done to
// create `c_hash`.
#include <libcuckoo-c/cuckoo_table_template.cc>
