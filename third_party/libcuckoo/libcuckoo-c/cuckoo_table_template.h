// To create an interface instance of the cuckoo table for a certain key-value
// type, you must define the following constants in your header file:
//
// Name of table
// #define CUCKOO_TABLE_NAME ___
//
// Type of key
// #define CUCKOO_KEY_TYPE ___
//
// Type of mapped value
// #define CUCKOO_MAPPED_TYPE ___
//
// Then, include this template file, which will fill in the interface
// definition.  If you are including multiple different table interfaces in the
// same compilation unit, make sure to undefine the three symbols above using
// the `#undef` macro.
//
// EXCEPTION SAFETY NOTE:
// Assuming no user defined data types, hash functions, or equality functions
// throw exceptions, the only exception that could be thrown by the table is
// std::bad_alloc whenever we allocate memory. This could occur when
// initializing objects, like the hashtable, a locked_table object, or an
// iterator. Or it could occur when we resize the table. In this latter case,
// all functions that could trigger a resize (upsert, insert, insert_or_assign,
// rehash, reserve, locked_table::insert, locked_table::rehash,
// locked_table::reserve) could throw. For all functions that allocate memory,
// we catch std::bad_alloc and set errno to ENOMEM.

// Helper macros, we take care of undefining these
#define PASTE2(a, b) a##b
#define PASTE(a, b) PASTE2(a, b)
#define CUCKOO(a) PASTE(CUCKOO_TABLE_NAME, a)

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

// Abstract type of the cuckoo table
struct CUCKOO_TABLE_NAME;
typedef struct CUCKOO_TABLE_NAME CUCKOO_TABLE_NAME;

// Typedefs for the key and value types
#define CUCKOO_KEY_ALIAS CUCKOO(_key_type)
typedef CUCKOO_KEY_TYPE CUCKOO_KEY_ALIAS;
#define CUCKOO_MAPPED_ALIAS CUCKOO(_mapped_type)
typedef CUCKOO_MAPPED_TYPE CUCKOO_MAPPED_ALIAS;

// Constructs a new table allocated to store at least `n` elements. Uses a
// default hash function, equality function, and allocator. There is no minimum
// load factor or maximum hashpower.
CUCKOO_TABLE_NAME *CUCKOO(_init)(size_t n);

// Reads in a table serialized in the file `fp` and constructs a new table.
// Uses a default hash function, equality function, and allocator. There is no
// minimum load factor or maximum hashpower. This will only work if the table
// types are POD, which should always be the case in a C program.
CUCKOO_TABLE_NAME *CUCKOO(_read)(FILE *fp);

// Destroys the given table
void CUCKOO(_free)(CUCKOO_TABLE_NAME *tbl);

// hashpower
size_t CUCKOO(_hashpower)(const CUCKOO_TABLE_NAME *tbl);

// bucket_count
size_t CUCKOO(_bucket_count)(const CUCKOO_TABLE_NAME *tbl);

// empty
bool CUCKOO(_empty)(const CUCKOO_TABLE_NAME *tbl);

// size
size_t CUCKOO(_size)(const CUCKOO_TABLE_NAME *tbl);

// capacity
size_t CUCKOO(_capacity)(const CUCKOO_TABLE_NAME *tbl);

// load_factor
double CUCKOO(_load_factor)(const CUCKOO_TABLE_NAME *tbl);

// find_fn
bool CUCKOO(_find_fn)(const CUCKOO_TABLE_NAME *tbl, const CUCKOO_KEY_ALIAS *key,
                      void (*fn)(const CUCKOO_MAPPED_ALIAS *));

// update_fn
bool CUCKOO(_update_fn)(CUCKOO_TABLE_NAME *tbl, const CUCKOO_KEY_ALIAS *key,
                        void (*fn)(CUCKOO_MAPPED_ALIAS *));

// upsert
bool CUCKOO(_upsert)(CUCKOO_TABLE_NAME *tbl, const CUCKOO_KEY_ALIAS *key,
                     void (*fn)(CUCKOO_MAPPED_ALIAS *),
                     const CUCKOO_MAPPED_ALIAS *val);

// erase_fn
bool CUCKOO(_erase_fn)(CUCKOO_TABLE_NAME *tbl, const CUCKOO_KEY_ALIAS *key,
                       bool (*fn)(CUCKOO_MAPPED_ALIAS *));

// find
bool CUCKOO(_find)(const CUCKOO_TABLE_NAME *tbl, const CUCKOO_KEY_ALIAS *key,
                   CUCKOO_MAPPED_ALIAS *val);

// contains
bool CUCKOO(_contains)(const CUCKOO_TABLE_NAME *tbl,
                       const CUCKOO_KEY_ALIAS *key);

// update
bool CUCKOO(_update)(CUCKOO_TABLE_NAME *tbl, const CUCKOO_KEY_ALIAS *key,
                     const CUCKOO_MAPPED_ALIAS *val);

// insert
bool CUCKOO(_insert)(CUCKOO_TABLE_NAME *tbl, const CUCKOO_KEY_ALIAS *key,
                     const CUCKOO_MAPPED_ALIAS *val);

// insert_or_assign
bool CUCKOO(_insert_or_assign)(CUCKOO_TABLE_NAME *tbl,
                               const CUCKOO_KEY_ALIAS *key,
                               const CUCKOO_MAPPED_ALIAS *val);

// erase
bool CUCKOO(_erase)(CUCKOO_TABLE_NAME *tbl, const CUCKOO_KEY_ALIAS *key);

// rehash
bool CUCKOO(_rehash)(CUCKOO_TABLE_NAME *tbl, size_t n);

// reserve
bool CUCKOO(_reserve)(CUCKOO_TABLE_NAME *tbl, size_t n);

// clear
void CUCKOO(_clear)(CUCKOO_TABLE_NAME *tbl);

// Abstract type of the locked table
#define CUCKOO_LOCKED_TABLE CUCKOO(_locked_table)
#define CUCKOO_LT(a) PASTE(CUCKOO_LOCKED_TABLE, a)
struct CUCKOO_LOCKED_TABLE;
typedef struct CUCKOO_LOCKED_TABLE CUCKOO_LOCKED_TABLE;

// lock_table -- this is the only way to construct a locked table
CUCKOO_LOCKED_TABLE *CUCKOO(_lock_table)(CUCKOO_TABLE_NAME *tbl);

// free locked_table
void CUCKOO_LT(_free)(CUCKOO_LOCKED_TABLE *ltbl);

// locked_table::unlock
void CUCKOO_LT(_unlock)(CUCKOO_LOCKED_TABLE *ltbl);

// locked_table::is_active
bool CUCKOO_LT(_is_active)(const CUCKOO_LOCKED_TABLE *ltbl);

// locked_table::hashpower
size_t CUCKOO_LT(_hashpower)(const CUCKOO_LOCKED_TABLE *ltbl);

// locked_table::bucket_count
size_t CUCKOO_LT(_bucket_count)(const CUCKOO_LOCKED_TABLE *ltbl);

// locked_table::empty
bool CUCKOO_LT(_empty)(const CUCKOO_LOCKED_TABLE *ltbl);

// locked_table::size
size_t CUCKOO_LT(_size)(const CUCKOO_LOCKED_TABLE *ltbl);

// locked_table::capacity
size_t CUCKOO_LT(_capacity)(const CUCKOO_LOCKED_TABLE *ltbl);

// locked_table::load_factor
double CUCKOO_LT(_load_factor)(const CUCKOO_LOCKED_TABLE *ltbl);

// Abstract type of the iterators
#define CUCKOO_ITERATOR CUCKOO(_iterator)
#define CUCKOO_IT(a) PASTE(CUCKOO_ITERATOR, a)
struct CUCKOO_ITERATOR;
typedef struct CUCKOO_ITERATOR CUCKOO_ITERATOR;

#define CUCKOO_CONST_ITERATOR CUCKOO(_const_iterator)
#define CUCKOO_CONST_IT(a) PASTE(CUCKOO_CONST_ITERATOR, a)
struct CUCKOO_CONST_ITERATOR;
typedef struct CUCKOO_CONST_ITERATOR CUCKOO_CONST_ITERATOR;

// begin and end are the only way to construct iterators

// locked_table::begin
CUCKOO_ITERATOR *CUCKOO_LT(_begin)(CUCKOO_LOCKED_TABLE *ltbl);

// locked_table::cbegin
CUCKOO_CONST_ITERATOR *CUCKOO_LT(_cbegin)(const CUCKOO_LOCKED_TABLE *ltbl);

// locked_table::end
CUCKOO_ITERATOR *CUCKOO_LT(_end)(CUCKOO_LOCKED_TABLE *ltbl);

// locked_table::cend
CUCKOO_CONST_ITERATOR *CUCKOO_LT(_cend)(const CUCKOO_LOCKED_TABLE *ltbl);

// free iterator
void CUCKOO_IT(_free)(CUCKOO_ITERATOR *it);

// free const iterator
void CUCKOO_CONST_IT(_free)(CUCKOO_CONST_ITERATOR *it);

// locked_table::clear
void CUCKOO_LT(_clear)(CUCKOO_LOCKED_TABLE *ltbl);

// locked_table::insert -- passing NULL for the iterator will not save the
// position of the inserted element
bool CUCKOO_LT(_insert)(CUCKOO_LOCKED_TABLE *ltbl, const CUCKOO_KEY_ALIAS *key,
                        const CUCKOO_MAPPED_ALIAS *val, CUCKOO_ITERATOR *it);

// locked_table::erase -- passing NULL for the iterator will not save the
// position of the following element
void CUCKOO_LT(_erase_it)(CUCKOO_LOCKED_TABLE *ltbl, CUCKOO_ITERATOR *it,
                          CUCKOO_ITERATOR *nextit);

// locked_table::erase const iterator
void CUCKOO_LT(_erase_const_it)(CUCKOO_LOCKED_TABLE *ltbl,
                                CUCKOO_CONST_ITERATOR *it,
                                CUCKOO_ITERATOR *nextit);

// locked_table::erase
size_t CUCKOO_LT(_erase)(CUCKOO_LOCKED_TABLE *ltbl,
                         const CUCKOO_KEY_ALIAS *key);

// locked_table::find -- the iterator passed in cannot be NULL
void CUCKOO_LT(_find)(CUCKOO_LOCKED_TABLE *ltbl, const CUCKOO_KEY_ALIAS *key,
                      CUCKOO_ITERATOR *it);

void CUCKOO_LT(_find_const)(const CUCKOO_LOCKED_TABLE *ltbl,
                            const CUCKOO_KEY_ALIAS *key,
                            CUCKOO_CONST_ITERATOR *it);

// locked_table::rehash
void CUCKOO_LT(_rehash)(CUCKOO_LOCKED_TABLE *ltbl, size_t n);

// locked_table::reserve
void CUCKOO_LT(_reserve)(CUCKOO_LOCKED_TABLE *ltbl, size_t n);

// locked_table::write will serialize the table to the file `fp`. This will
// only work if the table types are POD, which should always be the case in a C
// program.
bool CUCKOO_LT(_write)(const CUCKOO_LOCKED_TABLE *ltbl, FILE *fp);

// iterator::copy assignment
void CUCKOO_IT(_set)(CUCKOO_ITERATOR *dst, CUCKOO_ITERATOR *src);
void CUCKOO_CONST_IT(_set)(CUCKOO_CONST_ITERATOR *dst,
                           CUCKOO_CONST_ITERATOR *src);

// iterator::set to beginning
void CUCKOO_LT(_set_begin)(CUCKOO_LOCKED_TABLE *ltbl, CUCKOO_ITERATOR *it);
void CUCKOO_LT(_set_cbegin)(const CUCKOO_LOCKED_TABLE *ltbl,
                            CUCKOO_CONST_ITERATOR *it);

// iterator::set to end
void CUCKOO_LT(_set_end)(CUCKOO_LOCKED_TABLE *ltbl, CUCKOO_ITERATOR *it);
void CUCKOO_LT(_set_cend)(const CUCKOO_LOCKED_TABLE *ltbl,
                          CUCKOO_CONST_ITERATOR *it);

// iterator::equality
bool CUCKOO_IT(_equal)(CUCKOO_ITERATOR *it1, CUCKOO_ITERATOR *it2);
bool CUCKOO_CONST_IT(_equal)(CUCKOO_CONST_ITERATOR *it1,
                             CUCKOO_CONST_ITERATOR *it2);

// iterator::dereference key
const CUCKOO_KEY_ALIAS *CUCKOO_IT(_key)(CUCKOO_ITERATOR *it);
const CUCKOO_KEY_ALIAS *CUCKOO_CONST_IT(_key)(CUCKOO_CONST_ITERATOR *it);

// iterator::dereference mapped
CUCKOO_MAPPED_ALIAS *CUCKOO_IT(_mapped)(CUCKOO_ITERATOR *it);
const CUCKOO_MAPPED_ALIAS *CUCKOO_CONST_IT(_mapped)(CUCKOO_CONST_ITERATOR *it);

// iterator::increment
void CUCKOO_IT(_increment)(CUCKOO_ITERATOR *it);
void CUCKOO_CONST_IT(_increment)(CUCKOO_CONST_ITERATOR *it);

// iterator::decrement
void CUCKOO_IT(_decrement)(CUCKOO_ITERATOR *it);
void CUCKOO_CONST_IT(_decrement)(CUCKOO_CONST_ITERATOR *it);

// #undef the helper macros we defined
#undef PASTE
#undef PASTE2
#undef CUCKOO
#undef CUCKOO_KEY_ALIAS
#undef CUCKOO_MAPPED_ALIAS
#undef CUCKOO_LOCKED_TABLE
#undef CUCKOO_LT
#undef CUCKOO_ITERATOR
#undef CUCKOO_IT
#undef CUCKOO_CONST_ITERATOR
#undef CUCKOO_CONST_IT
