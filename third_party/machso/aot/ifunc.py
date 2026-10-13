"""Conservatively evaluate supported glibc IFUNC resolvers at compilation time."""
from capstone import Cs, CS_ARCH_ARM64, CS_MODE_LITTLE_ENDIAN
from capstone.arm64 import ARM64_OP_REG, ARM64_OP_IMM, ARM64_OP_MEM


class Resolver:
    RTLD_RO = 0x10000000

    def __init__(self, elf):
        self.elf = elf
        self.cs = Cs(CS_ARCH_ARM64, CS_MODE_LITTLE_ENDIAN)
        self.cs.detail = True
        self.relocations = {r[0]: r for r in elf.relocs}

    def memory(self, address, size):
        if self.RTLD_RO <= address < self.RTLD_RO+0x10000:
            # Baseline arm64: no Linux CPU extension dispatch, 64-byte cache lines.
            return {24: 16384, 120: 64}.get(address-self.RTLD_RO, 0)
        relocation = self.relocations.get(address)
        if relocation:
            _, kind, index, addend = relocation
            if kind == 1027:
                return addend
            symbol = self.elf.symbols[index]
            if symbol['name'] == '_rtld_global_ro':
                return self.RTLD_RO
            if symbol['shndx']:
                return symbol['value']+addend
            raise ValueError(f'IFUNC reads unsupported import {symbol["name"]}')
        return int.from_bytes(self.elf.read(address, size), 'little')

    def resolve(self, address):
        registers = {f'x{i}': 0 for i in range(31)}
        registers['sp'] = 0x20000000
        zero, carry = False, False

        def get(reg):
            name = self.cs.reg_name(reg)
            name = {'fp': 'x29', 'lr': 'x30'}.get(name, name)
            if name in ('xzr', 'wzr'):
                return 0
            if name.startswith('w'):
                return registers['x'+name[1:]] & 0xffffffff
            return registers[name]

        def put(reg, value):
            name = self.cs.reg_name(reg)
            name = {'fp': 'x29', 'lr': 'x30'}.get(name, name)
            if name in ('xzr', 'wzr'):
                return
            mask = 0xffffffff if name.startswith('w') else 0xffffffffffffffff
            registers['x'+name[1:] if name.startswith('w') else name] = value & mask

        def value(op):
            if op.type == ARM64_OP_REG:
                result = get(op.reg)
            elif op.type == ARM64_OP_IMM:
                result = op.imm
            else:
                raise ValueError('unsupported resolver operand')
            if op.shift.value:
                # Resolver instructions used here only shift left in operands.
                result <<= op.shift.value
            return result

        def condition(name):
            if name == 'eq': return zero
            if name == 'ne': return not zero
            if name in ('hs', 'cs'): return carry
            if name in ('lo', 'cc'): return not carry
            raise ValueError(f'unsupported IFUNC condition {name}')

        pc = address
        for _ in range(500):
            ins = next(self.cs.disasm(self.elf.read(pc, 4), pc))
            ops, mnemonic = ins.operands, ins.mnemonic
            next_pc = pc+4
            if mnemonic == 'adrp' or mnemonic == 'adr':
                put(ops[0].reg, ops[1].imm)
            elif mnemonic in ('ldr', 'ldrb', 'ldrh'):
                mem = ops[1].mem
                if ops[1].type != ARM64_OP_MEM or mem.index:
                    raise ValueError('unsupported IFUNC indexed load')
                size = {'ldrb': 1, 'ldrh': 2}.get(mnemonic, 4 if ins.reg_name(ops[0].reg).startswith('w') else 8)
                put(ops[0].reg, self.memory(get(mem.base)+mem.disp, size))
            elif mnemonic == 'mov':
                put(ops[0].reg, value(ops[1]))
            elif mnemonic in ('add', 'sub', 'and', 'eor', 'lsr'):
                a, b = value(ops[1]), value(ops[2])
                result = {'add': lambda: a+b, 'sub': lambda: a-b,
                          'and': lambda: a & b, 'eor': lambda: a ^ b,
                          'lsr': lambda: a >> b}[mnemonic]()
                put(ops[0].reg, result)
            elif mnemonic in ('cmp', 'tst'):
                a, b = value(ops[0]), value(ops[1])
                zero = (a == b) if mnemonic == 'cmp' else (a & b) == 0
                carry = a >= b
            elif mnemonic == 'csel':
                name = ins.op_str.rsplit(', ', 1)[1]
                put(ops[0].reg, value(ops[1] if condition(name) else ops[2]))
            elif mnemonic in ('tbz', 'tbnz'):
                test = bool(value(ops[0]) & (1 << ops[1].imm))
                if test == (mnemonic == 'tbnz'):
                    next_pc = ops[2].imm
            elif mnemonic in ('cbz', 'cbnz'):
                if (value(ops[0]) == 0) == (mnemonic == 'cbz'):
                    next_pc = ops[1].imm
            elif mnemonic.startswith('b.'):
                if condition(mnemonic[2:]): next_pc = ops[0].imm
            elif mnemonic == 'b':
                next_pc = ops[0].imm
            elif mnemonic == 'ret':
                result = registers['x0']
                if not any(p[1] & 1 and p[3] <= result < p[3]+p[5] for p in self.elf.loads):
                    raise ValueError('IFUNC did not return an executable address')
                return result
            elif mnemonic in ('stp', 'ldp', 'nop'):
                # The supported resolver prologues only save/restore x29 and x30.
                if mnemonic != 'nop' and not all(ins.reg_name(o.reg) in ('x29', 'x30', 'fp', 'lr') for o in ops[:2]):
                    raise ValueError('unsupported IFUNC stack operation')
            else:
                raise ValueError(f'unsupported IFUNC instruction at {pc:#x}: {mnemonic} {ins.op_str}')
            pc = next_pc
        raise ValueError('IFUNC instruction budget exceeded')
