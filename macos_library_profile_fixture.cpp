#include <cstdint>

#ifndef PROFILE_VARIANT
#define PROFILE_VARIANT 1
#endif

// Use a separate Mach-O section, like the blobs in a converted SDK dylib.
__attribute__((
    section("__DATA,__gpu_fixture"))) static const unsigned char bytes[] = {
    0x50, 0xed, 0x55, 0xba, PROFILE_VARIANT, 6, 7, 8};
static unsigned char zero_fill[4096];

extern "C" const unsigned char *profile_bytes() { return bytes; }
extern "C" unsigned char *profile_zero_fill() { return zero_fill; }
