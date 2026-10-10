"""Extract glibc tunable defaults from the matching ARM64 loader binary."""
import struct

from capstone import Cs, CS_ARCH_ARM64, CS_MODE_LITTLE_ENDIAN
from capstone.arm64 import ARM64_OP_IMM, ARM64_OP_MEM


def extract(elf):
    symbol = next((s for s in elf.symbols if s['name'] == '__tunable_get_val' and s['shndx']), None)
    if symbol is None:
        raise ValueError('loader does not export __tunable_get_val')
    decoder = Cs(CS_ARCH_ARM64, CS_MODE_LITTLE_ENDIAN)
    decoder.detail = True
    instructions = list(decoder.disasm(elf.read(symbol['value'], symbol['size']), symbol['value']))
    stride = None
    table = None
    pages = {}
    offsets = {}
    for instruction in instructions:
        operands = instruction.operands
        if instruction.mnemonic == 'ubfiz' and operands[2].type == ARM64_OP_IMM:
            stride = 1 << operands[2].imm
        if instruction.mnemonic == 'adrp':
            pages[operands[0].reg] = operands[1].imm
        if instruction.mnemonic == 'add' and len(operands) == 3 and operands[2].type == ARM64_OP_IMM:
            if operands[1].reg in pages:
                table = pages[operands[1].reg]+operands[2].imm
        if instruction.mnemonic in ('ldr', 'ldrb') and operands[1].type == ARM64_OP_MEM:
            register = instruction.reg_name(operands[0].reg)
            if instruction.mnemonic == 'ldrb':
                offsets['initialized'] = operands[1].mem.disp
            elif register.startswith('w'):
                offsets.setdefault('type', operands[1].mem.disp)
            elif register.startswith('x'):
                offsets.setdefault('value', operands[1].mem.disp)
    if stride != 128 or table is None or offsets != {'type': 48, 'value': 88, 'initialized': 104}:
        raise ValueError('unsupported glibc tunable table layout')
    entries = []
    for index in range(256):
        record = elf.read(table+index*stride, stride)
        name = record[:offsets['type']].split(b'\0', 1)[0]
        if not name.startswith(b'glibc.'):
            break
        kind = struct.unpack_from('<I', record, offsets['type'])[0]
        value, length = struct.unpack_from('<QQ', record, offsets['value'])
        if kind not in (0, 1, 2, 3) or record[offsets['initialized']]:
            raise ValueError('unsupported initialized glibc tunable')
        if kind == 3 and (value or length):
            raise ValueError('nonempty string tunable defaults need pointer relocation')
        entries.append({'name': name.decode('ascii'), 'type': kind, 'value': value})
    if not entries:
        raise ValueError('loader has no tunable defaults')
    return entries


def native_code(entries):
    rows = ',\n'.join(f'    {{{item["type"]}, UINT64_C({item["value"]})}}' for item in entries)
    return '''static const struct {unsigned type; uint64_t value;} aot_tunables[] = {
'''+rows+'''
};
void aot_import___tunable_get_val(int id, void *value, void *callback) {
    (void)callback; /* Defaults are not initialized by an environment override. */
    if (id<0 || (size_t)id>=sizeof(aot_tunables)/sizeof(aot_tunables[0]) || !value)
        __builtin_trap();
    if (aot_tunables[id].type==0) *(uint32_t *)value=(uint32_t)aot_tunables[id].value;
    else if (aot_tunables[id].type==3) {
        ((uintptr_t *)value)[0]=0; ((uintptr_t *)value)[1]=0;
    } else *(uint64_t *)value=aot_tunables[id].value;
}
'''
