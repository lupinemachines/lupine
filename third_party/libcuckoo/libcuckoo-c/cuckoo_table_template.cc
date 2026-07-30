// To create an implementation of the cuckoo table for a certain key-value type,
// include your header file in an extern "C" file as follows:
//
// extern "C" {
//     #include "interface_header.h"
// }
//
// Then include this template file

#include <cerrno>
#include <cstdio>
#include <memory>
#include <utility>

#include <libcuckoo/cuckoohash_map.hh>

// Helper macros, we take care of undefining these
#define PASTE2(a, b) a##b
#define PASTE(a, b) PASTE2(a, b)
#define CUCKOO(a) PASTE(CUCKOO_TABLE_NAME, a)

#ifdef __cplusplus
extern "C" {
#endif

typedef libcuckoo::cuckoohash_map<CUCKOO_KEY_TYPE, CUCKOO_MAPPED_TYPE> tbl_t;

struct CUCKOO_TABLE_NAME {
  tbl_t t;
  CUCKOO_TABLE_NAME(size_t n) : t(n) {}
};
typedef struct CUCKOO_TABLE_NAME CUCKOO_TABLE_NAME;

#define CUCKOO_KEY_ALIAS CUCKOO(_key_type)
typedef CUCKOO_KEY_TYPE CUCKOO_KEY_ALIAS;
#define CUCKOO_MAPPED_ALIAS CUCKOO(_mapped_type)
typedef CUCKOO_MAPPED_TYPE CUCKOO_MAPPED_ALIAS;

CUCKOO_TABLE_NAME *CUCKOO(_init)(size_t n) {
  CUCKOO_TABLE_NAME *tbl;
  try {
    tbl = new CUCKOO_TABLE_NAME(n);
  } catch (std::bad_alloc &) {
    errno = ENOMEM;
    return NULL;
  }
  tbl->t.minimum_load_factor(0);
  tbl->t.maximum_hashpower(libcuckoo::NO_MAXIMUM_HASHPOWER);
  return tbl;
}

CUCKOO_TABLE_NAME *CUCKOO(_read)(FILE *fp) {
  size_t tbl_size;
  if (!fread(&tbl_size, sizeof(size_t), 1, fp)) {
    return NULL;
  }
  CUCKOO_TABLE_NAME *tbl = NULL;
  try {
    tbl = new CUCKOO_TABLE_NAME(tbl_size);
  } catch (std::bad_alloc &) {
    errno = ENOMEM;
    return NULL;
  }
  CUCKOO_KEY_ALIAS key;
  CUCKOO_MAPPED_ALIAS mapped;
  for (size_t i = 0; i < tbl_size; ++i) {
    if (!fread(&key, sizeof(CUCKOO_KEY_ALIAS), 1, fp)) {
      delete tbl;
      return NULL;
    }
    if (!fread(&mapped, sizeof(CUCKOO_MAPPED_ALIAS), 1, fp)) {
      delete tbl;
      return NULL;
    }
    try {
      tbl->t.insert(key, mapped);
    } catch (std::bad_alloc &) {
      delete tbl;
      errno = ENOMEM;
      return NULL;
    }
  }
  return tbl;
}

void CUCKOO(_free)(CUCKOO_TABLE_NAME *tbl) { delete tbl; }

// hashpower
size_t CUCKOO(_hashpower)(const CUCKOO_TABLE_NAME *tbl) {
  return tbl->t.hashpower();
}

// bucket_count
size_t CUCKOO(_bucket_count)(const CUCKOO_TABLE_NAME *tbl) {
  return tbl->t.bucket_count();
}

// empty
bool CUCKOO(_empty)(const CUCKOO_TABLE_NAME *tbl) { return tbl->t.empty(); }

// size
size_t CUCKOO(_size)(const CUCKOO_TABLE_NAME *tbl) { return tbl->t.size(); }

// capacity
size_t CUCKOO(_capacity)(const CUCKOO_TABLE_NAME *tbl) {
  return tbl->t.capacity();
}

// load_factor
double CUCKOO(_load_factor)(const CUCKOO_TABLE_NAME *tbl) {
  return tbl->t.load_factor();
}

// find_fn
bool CUCKOO(_find_fn)(const CUCKOO_TABLE_NAME *tbl, const CUCKOO_KEY_ALIAS *key,
                      void (*fn)(const CUCKOO_MAPPED_ALIAS *)) {
  return tbl->t.find_fn(*key, [&fn](const CUCKOO_MAPPED_ALIAS &v) { fn(&v); });
}

// update_fn
bool CUCKOO(_update_fn)(CUCKOO_TABLE_NAME *tbl, const CUCKOO_KEY_ALIAS *key,
                        void (*fn)(CUCKOO_MAPPED_ALIAS *)) {
  return tbl->t.update_fn(*key, [&fn](CUCKOO_MAPPED_ALIAS &v) { fn(&v); });
}

// upsert
bool CUCKOO(_upsert)(CUCKOO_TABLE_NAME *tbl, const CUCKOO_KEY_ALIAS *key,
                     void (*fn)(CUCKOO_MAPPED_ALIAS *),
                     const CUCKOO_MAPPED_ALIAS *value) {
  try {
    return tbl->t.upsert(*key, [&fn](CUCKOO_MAPPED_ALIAS &v) { fn(&v); },
                         *value);
  } catch (std::bad_alloc &) {
    errno = ENOMEM;
    return false;
  }
}

// erase_fn
bool CUCKOO(_erase_fn)(CUCKOO_TABLE_NAME *tbl, const CUCKOO_KEY_ALIAS *key,
                       bool (*fn)(CUCKOO_MAPPED_ALIAS *)) {
  return tbl->t.erase_fn(
      *key, [&fn](CUCKOO_MAPPED_ALIAS &v) -> bool { return fn(&v); });
}

// find
bool CUCKOO(_find)(const CUCKOO_TABLE_NAME *tbl, const CUCKOO_KEY_ALIAS *key,
                   CUCKOO_MAPPED_ALIAS *val) {
  return tbl->t.find(*key, *val);
}

// contains
bool CUCKOO(_contains)(const CUCKOO_TABLE_NAME *tbl,
                       const CUCKOO_KEY_ALIAS *key) {
  return tbl->t.contains(*key);
}

// update
bool CUCKOO(_update)(CUCKOO_TABLE_NAME *tbl, const CUCKOO_KEY_ALIAS *key,
                     const CUCKOO_MAPPED_ALIAS *val) {
  return tbl->t.update(*key, *val);
}

// insert
bool CUCKOO(_insert)(CUCKOO_TABLE_NAME *tbl, const CUCKOO_KEY_ALIAS *key,
                     const CUCKOO_MAPPED_ALIAS *val) {
  try {
    return tbl->t.insert(*key, *val);
  } catch (std::bad_alloc &) {
    errno = ENOMEM;
    return false;
  }
}

// insert_or_assign
bool CUCKOO(_insert_or_assign)(CUCKOO_TABLE_NAME *tbl,
                               const CUCKOO_KEY_ALIAS *key,
                               const CUCKOO_MAPPED_ALIAS *val) {
  try {
    return tbl->t.insert_or_assign(*key, *val);
  } catch (std::bad_alloc &) {
    errno = ENOMEM;
    return false;
  }
}

// erase
bool CUCKOO(_erase)(CUCKOO_TABLE_NAME *tbl, const CUCKOO_KEY_ALIAS *key) {
  return tbl->t.erase(*key);
}

// rehash
bool CUCKOO(_rehash)(CUCKOO_TABLE_NAME *tbl, size_t n) {
  try {
    return tbl->t.rehash(n);
  } catch (std::bad_alloc &) {
    errno = ENOMEM;
    return false;
  }
}

// reserve
bool CUCKOO(_reserve)(CUCKOO_TABLE_NAME *tbl, size_t n) {
  try {
    return tbl->t.reserve(n);
  } catch (std::bad_alloc &) {
    errno = ENOMEM;
    return false;
  }
}

// clear
void CUCKOO(_clear)(CUCKOO_TABLE_NAME *tbl) { tbl->t.clear(); }

#define CUCKOO_LOCKED_TABLE CUCKOO(_locked_table)
#define CUCKOO_LT(a) PASTE(CUCKOO_LOCKED_TABLE, a)
struct CUCKOO_LOCKED_TABLE {
  tbl_t::locked_table lt;

  CUCKOO_LOCKED_TABLE(CUCKOO_TABLE_NAME *tbl)
      : lt(std::move(tbl->t.lock_table())) {}
};
typedef struct CUCKOO_LOCKED_TABLE CUCKOO_LOCKED_TABLE;

// lock_table -- this is the only way to construct a locked table
CUCKOO_LOCKED_TABLE *CUCKOO(_lock_table)(CUCKOO_TABLE_NAME *tbl) {
  try {
    return new CUCKOO_LOCKED_TABLE(tbl);
  } catch (std::bad_alloc &) {
    errno = ENOMEM;
    return NULL;
  }
}

// free locked_table
void CUCKOO_LT(_free)(CUCKOO_LOCKED_TABLE *ltbl) { delete ltbl; }

// locked_table::unlock
void CUCKOO_LT(_unlock)(CUCKOO_LOCKED_TABLE *ltbl) { ltbl->lt.unlock(); }

// locked_table::is_active
bool CUCKOO_LT(_is_active)(const CUCKOO_LOCKED_TABLE *ltbl) {
  return ltbl->lt.is_active();
}

// locked_table::hashpower
size_t CUCKOO_LT(_hashpower)(const CUCKOO_LOCKED_TABLE *ltbl) {
  return ltbl->lt.hashpower();
}

// locked_table::bucket_count
size_t CUCKOO_LT(_bucket_count)(const CUCKOO_LOCKED_TABLE *ltbl) {
  return ltbl->lt.bucket_count();
}

// locked_table::empty
bool CUCKOO_LT(_empty)(const CUCKOO_LOCKED_TABLE *ltbl) {
  return ltbl->lt.empty();
}

// locked_table::size
size_t CUCKOO_LT(_size)(const CUCKOO_LOCKED_TABLE *ltbl) {
  return ltbl->lt.size();
}

// locked_table::capacity
size_t CUCKOO_LT(_capacity)(const CUCKOO_LOCKED_TABLE *ltbl) {
  return ltbl->lt.capacity();
}

// locked_table::load_factor
double CUCKOO_LT(_load_factor)(const CUCKOO_LOCKED_TABLE *ltbl) {
  return ltbl->lt.load_factor();
}

#define CUCKOO_ITERATOR CUCKOO(_iterator)
#define CUCKOO_IT(a) PASTE(CUCKOO_ITERATOR, a)
struct CUCKOO_ITERATOR {
  tbl_t::locked_table::iterator it;
  CUCKOO_ITERATOR(tbl_t::locked_table::iterator i) : it(i) {}
};
typedef struct CUCKOO_ITERATOR CUCKOO_ITERATOR;

#define CUCKOO_CONST_ITERATOR CUCKOO(_const_iterator)
#define CUCKOO_CONST_IT(a) PASTE(CUCKOO_CONST_ITERATOR, a)
struct CUCKOO_CONST_ITERATOR {
  tbl_t::locked_table::const_iterator it;
  CUCKOO_CONST_ITERATOR(tbl_t::locked_table::const_iterator i) : it(i) {}
};
typedef struct CUCKOO_CONST_ITERATOR CUCKOO_CONST_ITERATOR;

// the following four functions are the only way to construct iterators

// locked_table::begin
CUCKOO_ITERATOR *CUCKOO_LT(_begin)(CUCKOO_LOCKED_TABLE *ltbl) {
  try {
    return new CUCKOO_ITERATOR(ltbl->lt.begin());
  } catch (std::bad_alloc &) {
    errno = ENOMEM;
    return NULL;
  }
}

// locked_table::cbegin
CUCKOO_CONST_ITERATOR *CUCKOO_LT(_cbegin)(const CUCKOO_LOCKED_TABLE *ltbl) {
  try {
    return new CUCKOO_CONST_ITERATOR(ltbl->lt.cbegin());
  } catch (std::bad_alloc &) {
    errno = ENOMEM;
    return NULL;
  }
}

// locked_table::end
CUCKOO_ITERATOR *CUCKOO_LT(_end)(CUCKOO_LOCKED_TABLE *ltbl) {
  try {
    return new CUCKOO_ITERATOR(ltbl->lt.end());
  } catch (std::bad_alloc &) {
    errno = ENOMEM;
    return NULL;
  }
}

// locked_table::cend
CUCKOO_CONST_ITERATOR *CUCKOO_LT(_cend)(const CUCKOO_LOCKED_TABLE *ltbl) {
  try {
    return new CUCKOO_CONST_ITERATOR(ltbl->lt.cend());
  } catch (std::bad_alloc &) {
    errno = ENOMEM;
    return NULL;
  }
}

// free iterator
void CUCKOO_IT(_free)(CUCKOO_ITERATOR *it) { delete it; }
void CUCKOO_CONST_IT(_free)(CUCKOO_CONST_ITERATOR *it) { delete it; }

// locked_table::clear
void CUCKOO_LT(_clear)(CUCKOO_LOCKED_TABLE *ltbl) { ltbl->lt.clear(); }

// locked_table::insert -- passing NULL for the iterator will not save the
// position of the inserted element
bool CUCKOO_LT(_insert)(CUCKOO_LOCKED_TABLE *ltbl, const CUCKOO_KEY_ALIAS *key,
                        const CUCKOO_MAPPED_ALIAS *val, CUCKOO_ITERATOR *it) {
  std::pair<tbl_t::locked_table::iterator, bool> ret;
  try {
    ret = ltbl->lt.insert(*key, *val);
  } catch (std::bad_alloc &) {
    errno = ENOMEM;
    return false;
  }
  if (it != NULL) {
    it->it = ret.first;
  }
  return ret.second;
}

// locked_table::erase -- passing NULL for the iterator will not save the
// position of the following element
void CUCKOO_LT(_erase_it)(CUCKOO_LOCKED_TABLE *ltbl, CUCKOO_ITERATOR *it,
                          CUCKOO_ITERATOR *nextit) {
  auto ret = ltbl->lt.erase(it->it);
  if (nextit != NULL) {
    nextit->it = ret;
  }
}

// locked_table::erase const iterator
void CUCKOO_LT(_erase_const_it)(CUCKOO_LOCKED_TABLE *ltbl,
                                CUCKOO_CONST_ITERATOR *it,
                                CUCKOO_ITERATOR *nextit) {
  auto ret = ltbl->lt.erase(it->it);
  if (nextit != NULL) {
    nextit->it = ret;
  }
}

// locked_table::erase
size_t CUCKOO_LT(_erase)(CUCKOO_LOCKED_TABLE *ltbl,
                         const CUCKOO_KEY_ALIAS *key) {
  return ltbl->lt.erase(*key);
}

// locked_table::find -- the iterator passed in cannot be NULL
void CUCKOO_LT(_find)(CUCKOO_LOCKED_TABLE *ltbl, const CUCKOO_KEY_ALIAS *key,
                      CUCKOO_ITERATOR *it) {
  it->it = ltbl->lt.find(*key);
}

void CUCKOO_LT(_find_const)(const CUCKOO_LOCKED_TABLE *ltbl,
                            const CUCKOO_KEY_ALIAS *key,
                            CUCKOO_CONST_ITERATOR *it) {
  it->it = ltbl->lt.find(*key);
}

// locked_table::rehash
void CUCKOO_LT(_rehash)(CUCKOO_LOCKED_TABLE *ltbl, size_t n) {
  try {
    ltbl->lt.rehash(n);
  } catch (std::bad_alloc &) {
    errno = ENOMEM;
  }
}

// locked_table::reserve
void CUCKOO_LT(_reserve)(CUCKOO_LOCKED_TABLE *ltbl, size_t n) {
  try {
    ltbl->lt.reserve(n);
  } catch (std::bad_alloc &) {
    errno = ENOMEM;
  }
}

// locked_table::write
bool CUCKOO_LT(_write)(const CUCKOO_LOCKED_TABLE *ltbl, FILE *fp) {
  size_t tbl_size = ltbl->lt.size();
  if (!fwrite(&tbl_size, sizeof(size_t), 1, fp)) {
    return false;
  }
  for (const auto &pair : ltbl->lt) {
    if (!fwrite(std::addressof(pair.first), sizeof(CUCKOO_KEY_ALIAS), 1, fp)) {
      return false;
    }
    if (!fwrite(std::addressof(pair.second), sizeof(CUCKOO_MAPPED_ALIAS), 1,
                fp)) {
      return false;
    }
  }
  return true;
}

// iterator::copy assignment
void CUCKOO_IT(_set)(CUCKOO_ITERATOR *dst, CUCKOO_ITERATOR *src) {
  dst->it = src->it;
}
void CUCKOO_CONST_IT(_set)(CUCKOO_CONST_ITERATOR *dst,
                           CUCKOO_CONST_ITERATOR *src) {
  dst->it = src->it;
}

// iterator::set to beginning
void CUCKOO_LT(_set_begin)(CUCKOO_LOCKED_TABLE *ltbl, CUCKOO_ITERATOR *it) {
  it->it = ltbl->lt.begin();
}
void CUCKOO_LT(_set_cbegin)(const CUCKOO_LOCKED_TABLE *ltbl,
                            CUCKOO_CONST_ITERATOR *it) {
  it->it = ltbl->lt.cbegin();
}

// iterator::set to end
void CUCKOO_LT(_set_end)(CUCKOO_LOCKED_TABLE *ltbl, CUCKOO_ITERATOR *it) {
  it->it = ltbl->lt.end();
}
void CUCKOO_LT(_set_cend)(const CUCKOO_LOCKED_TABLE *ltbl,
                          CUCKOO_CONST_ITERATOR *it) {
  it->it = ltbl->lt.cend();
}

// iterator::equality
bool CUCKOO_IT(_equal)(CUCKOO_ITERATOR *it1, CUCKOO_ITERATOR *it2) {
  return it1->it == it2->it;
}
bool CUCKOO_CONST_IT(_equal)(CUCKOO_CONST_ITERATOR *it1,
                             CUCKOO_CONST_ITERATOR *it2) {
  return it1->it == it2->it;
}

// iterator::dereference key
const CUCKOO_KEY_ALIAS *CUCKOO_IT(_key)(CUCKOO_ITERATOR *it) {
  return &it->it->first;
}
const CUCKOO_KEY_ALIAS *CUCKOO_CONST_IT(_key)(CUCKOO_CONST_ITERATOR *it) {
  return &it->it->first;
}

// iterator::dereference mapped
CUCKOO_MAPPED_ALIAS *CUCKOO_IT(_mapped)(CUCKOO_ITERATOR *it) {
  return &it->it->second;
}
const CUCKOO_MAPPED_ALIAS *CUCKOO_CONST_IT(_mapped)(CUCKOO_CONST_ITERATOR *it) {
  return &it->it->second;
}

// iterator::increment
void CUCKOO_IT(_increment)(CUCKOO_ITERATOR *it) { ++(it->it); }
void CUCKOO_CONST_IT(_increment)(CUCKOO_CONST_ITERATOR *it) { ++(it->it); }

// iterator::decrement
void CUCKOO_IT(_decrement)(CUCKOO_ITERATOR *it) { --(it->it); }
void CUCKOO_CONST_IT(_decrement)(CUCKOO_CONST_ITERATOR *it) { --(it->it); }

#ifdef __cplusplus
}
#endif

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
