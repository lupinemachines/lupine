.text
.p2align 2
.macro BRIDGE name, kind
.globl _elf_\name
_elf_\name:
    stp x29, x30, [sp, #-160]!
    mov x29, sp
    stp x0, x1, [sp, #16]
    stp x2, x3, [sp, #32]
    stp x4, x5, [sp, #48]
    stp x6, x7, [sp, #64]
    stp d0, d1, [sp, #80]
    stp d2, d3, [sp, #96]
    stp d4, d5, [sp, #112]
    stp d6, d7, [sp, #128]
    add x9, sp, #160
    str x9, [sp, #144]
    add x0, sp, #16
    mov w1, #\kind
    bl _elf_format
    ldp x29, x30, [sp], #160
    ret
.endm
BRIDGE snprintf, 0
BRIDGE printf, 1
BRIDGE fprintf, 2
BRIDGE __snprintf_chk, 3
BRIDGE __printf_chk, 4
BRIDGE __fprintf_chk, 5
