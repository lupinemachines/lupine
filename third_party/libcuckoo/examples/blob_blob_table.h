// This table was created to illustrate usage of the table with untyped blobs.
// See `int_str_table.h` for more comments regarding the semantics of creating
// header files for use with C programs.

#ifndef BLOB_BLOB_TABLE_H
#define BLOB_BLOB_TABLE_H

// In C, assignment is not defined for raw arrays, but it is defined for all
// structs, including those containing arrays. Thus, since the hashtable
// requires that assignment is defined for key and mapped types, we wrap the
// blobs in a struct.
typedef struct { char blob[8]; } key_blob;

typedef struct { char blob[255]; } mapped_blob;

#define CUCKOO_TABLE_NAME blob_blob_table
#define CUCKOO_KEY_TYPE key_blob
#define CUCKOO_MAPPED_TYPE mapped_blob

#include <libcuckoo-c/cuckoo_table_template.h>

#endif // BLOB_BLOB_TABLE_H
