#include <stdint.h>

#ifdef __APPLE__
int abi_preserves_host_x18(void *(*function)(void), void *expected) {
  uintptr_t before, after;
  __asm__ volatile("mov %0, x18" : "=r"(before) : : "memory");
  void *result = function();
  __asm__ volatile("mov %0, x18" : "=r"(after) : : "memory");
  return result == expected && before == after;
}
#else
int abi_scalar_mix(float f0, int i0, float f1, int i1, float f2, int i2,
                   float f3, int i3, float f4, int i4, float f5, int i5,
                   float f6, int i6, float f7, int i7, float f8, int i8,
                   float f9, int i9, float f10, int i10) {
  return (int)f0 + i0 + (int)f1 + i1 + (int)f2 + i2 + (int)f3 + i3 + (int)f4 +
         i4 + (int)f5 + i5 + (int)f6 + i6 + (int)f7 + i7 + (int)f8 + i8 +
         (int)f9 + i9 + (int)f10 + i10;
}

__asm__(".text\n"
        ".p2align 2\n"
        ".global abi_tls_x18\n.type abi_tls_x18,%function\n"
        "abi_tls_x18:\n mrs x18,tpidr_el0\n mov x0,x18\n ret\n"
        ".global abi_tls_x0\n.type abi_tls_x0,%function\n"
        "abi_tls_x0:\n mrs x0,tpidr_el0\n ret\n"
        ".global abi_value_x18\n.type abi_value_x18,%function\n"
        "abi_value_x18:\n mov x18,#42\n mov x0,x18\n ret\n");
#endif
