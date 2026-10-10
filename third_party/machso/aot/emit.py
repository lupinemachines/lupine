"""Emit a native object retaining original ELF code, with AOT platform lowering."""
import json
import struct
import re
from pathlib import Path
from capstone import Cs, CS_ARCH_ARM64, CS_MODE_LITTLE_ENDIAN
from capstone.arm64 import ARM64_OP_IMM
from machso import ELF, align
from aot.ifunc import Resolver

ELF_BASE = 0x400000
PRE_TCB = 0x1000
PUBLIC_WRAPPERS = {'snprintf', 'vsnprintf', 'printf', 'vprintf', 'fprintf', 'vfprintf',
                   '__snprintf_chk', '__printf_chk', '__fprintf_chk', 'syscall'}


def executable_sections(elf):
    header = struct.unpack_from('<HHIQQQIHHHHHH', elf.data, 16)
    if not header[5] or not header[11]:
        raise ValueError('AOT currently requires ELF section headers for code/data boundaries')
    sections = [struct.unpack_from('<IIQQQQIIQQ', elf.data, header[5]+i*header[10]) for i in range(header[11])]
    return [(s[3], elf.data[s[4]:s[4]+s[5]]) for s in sections if s[2] & 4 and s[1] == 1]


def scan(elf):
    cs = Cs(CS_ARCH_ARM64, CS_MODE_LITTLE_ENDIAN)
    cs.detail = True
    cs.skipdata = True
    instructions = []
    for address, data in executable_sections(elf):
        instructions.extend(cs.disasm(data, address))
    targets = set()
    for ins in instructions:
        if ins.mnemonic.startswith('b') or ins.mnemonic in ('cbz', 'cbnz', 'tbz', 'tbnz'):
            if ins.id and ins.operands and ins.operands[-1].type == ARM64_OP_IMM:
                targets.add(ins.operands[-1].imm)
    sites, x18 = [], []
    number = None
    for ins in instructions:
        if ins.address in targets:
            number = None
        if ins.mnemonic == 'mrs' and 'tpidr_el0' in ins.op_str:
            register = ins.op_str.split(',')[0]
            sites.append({'address': ins.address, 'kind': 'tls', 'register': int(register[1:])})
        elif ins.mnemonic == 'svc':
            sites.append({'address': ins.address, 'kind': 'syscall', 'number': number})
        if ins.id:
            _, writes = ins.regs_access()
            if any(ins.reg_name(r) in ('x8', 'w8') for r in writes):
                number = ins.operands[1].imm if ins.mnemonic == 'mov' and ins.operands[1].type == ARM64_OP_IMM else None
            if any(ins.reg_name(r) in ('x18', 'w18') for bank in ins.regs_access() for r in bank):
                writes_names = [ins.reg_name(r) for r in writes]
                record = {'address': ins.address, 'kind': 'x18', 'instruction': ins.mnemonic+' '+ins.op_str,
                          'writes': writes_names}
                x18.append(record)
                sites.append(record)
        if ins.mnemonic.startswith('b') or ins.mnemonic in ('ret', 'cbz', 'cbnz', 'tbz', 'tbnz'):
            number = None
    return sites, x18


def save_frame():
    lines = ['sub sp, sp, #800']
    for i in range(0, 30, 2): lines.append(f'stp x{i}, x{i+1}, [sp, #{i*8}]')
    lines += ['str x30, [sp, #240]', 'mrs x16, nzcv', 'str x16, [sp, #248]',
              'mrs x16, fpcr', 'str x16, [sp, #768]', 'mrs x16, fpsr', 'str x16, [sp, #776]']
    for i in range(0, 32, 2): lines.append(f'stp q{i}, q{i+1}, [sp, #{256+i*16}]')
    return lines


def restore_frame():
    lines = []
    for i in range(0, 32, 2): lines.append(f'ldp q{i}, q{i+1}, [sp, #{256+i*16}]')
    lines += ['ldr x16, [sp, #768]', 'msr fpcr, x16', 'ldr x16, [sp, #776]',
              'msr fpsr, x16', 'ldr x16, [sp, #248]', 'msr nzcv, x16']
    for i in range(0, 30, 2): lines.append(f'ldp x{i}, x{i+1}, [sp, #{i*8}]')
    lines += ['ldr x30, [sp, #240]', 'add sp, sp, #800']
    return lines


def lower_x18(site):
    """Replace Linux's x18 with a TLS slot; never execute guest code using host x18."""
    address = site['address']
    instruction = site['instruction']
    used = {int(n) for n in re.findall(r'\b[wx](\d+)\b', instruction)}
    scratch = [n for n in [16, 17, 15, 14, 13, 12, 11, 10, 9] if n not in used][:3]
    value_reg, pointer_reg, stack_reg = scratch
    lines = ['bl _aot_thread_pointer', 'str x0, [sp, #792]']
    # Recover the guest operands and condition flags after the native TLS call.
    for i in range(31):
        if i not in scratch and i != 18: lines.append(f'ldr x{i}, [sp, #{i*8}]')
    for i in range(0, 32, 2): lines.append(f'ldp q{i}, q{i+1}, [sp, #{256+i*16}]')
    lines += [f'ldr x{pointer_reg}, [sp, #248]', f'msr nzcv, x{pointer_reg}',
              f'ldr x{pointer_reg}, [sp, #792]', f'sub x{pointer_reg}, x{pointer_reg}, #0x1000',
              f'ldr x{value_reg}, [x{pointer_reg}]', f'add x{stack_reg}, sp, #800']
    rewritten = re.sub(r'\b([wx])18\b', lambda m: m[1]+str(value_reg), instruction)
    rewritten = re.sub(r'\bsp\b', f'x{stack_reg}', rewritten)
    mnemonic = instruction.split()[0]
    if mnemonic in ('blr', 'br', 'ret'):
        # Restore the caller's actual stack before entering the target. Keeping
        # the spill frame below SP would corrupt stack arguments and lose return
        # values. x16 is the AAPCS64 intra-procedure scratch register.
        lines += [f'str x{value_reg}, [sp, #128]']
        if mnemonic == 'blr':
            lines += [f'adrp x{pointer_reg}, Lelf_{address+4:x}@PAGE',
                      f'add x{pointer_reg}, x{pointer_reg}, Lelf_{address+4:x}@PAGEOFF',
                      f'str x{pointer_reg}, [sp, #240]']
        lines += [*restore_frame(), 'br x16']
        return lines, set()
    if mnemonic in ('cbz', 'cbnz', 'tbz', 'tbnz'):
        destination = int(rewritten.rsplit('#', 1)[1], 16)
        prefix = rewritten.rsplit('#', 1)[0]
        lines += [prefix+f'Laot_take_{address:x}', *restore_frame(), f'b Lelf_{address+4:x}',
                  f'Laot_take_{address:x}:', *restore_frame(), f'b Lelf_{destination:x}']
        return lines, {destination}
    targets = set()
    if mnemonic == 'adrp':
        destination = int(rewritten.rsplit('#', 1)[1], 16)
        rewritten = rewritten.rsplit('#', 1)[0]+f'Lelf_{destination:x}@PAGE'
        targets.add(destination)
    lines.append(rewritten)
    for name in site['writes']:
        normalized = {'fp': 'x29', 'lr': 'x30'}.get(name, name)
        if normalized.startswith(('x', 'w')) and normalized[1:].isdigit():
            index = int(normalized[1:])
            if index != 18: lines.append(f'str x{index}, [sp, #{index*8}]')
    if any(name in ('w18', 'x18') for name in site['writes']):
        lines.append(f'str x{value_reg}, [x{pointer_reg}]')
    lines += [f'mrs x{pointer_reg}, nzcv', f'str x{pointer_reg}, [sp, #248]',
              *restore_frame(), f'b Lelf_{address+4:x}']
    return lines, targets


def emit(source, directory, plan=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    elf = ELF(Path(source).read_bytes(), allow_tls=True)
    if not elf.loads or elf.loads[0][3] != 0:
        raise ValueError('AOT currently requires the first LOAD at virtual address zero')
    if plan is None and (len(elf.loads) != 2 or [p[1] for p in elf.loads] != [5, 6]):
        raise ValueError('standalone glibc profile requires RX at zero followed by RW')
    base = plan.base if plan else ELF_BASE
    segment_names = plan.segments if plan else ['__ELFTEXT', '__ELFDATA']
    core = plan.core if plan else True
    tls_offset = plan.tls_offset if plan else 16
    sites, x18 = scan(elf)
    resolver = Resolver(elf)
    ifuncs = {s['value']: resolver.resolve(s['value']) for s in elf.symbols if s['info'] & 15 == 10}
    for _, kind, _, addend in elf.relocs:
        if kind == 1032: ifuncs.setdefault(addend, resolver.resolve(addend))
    exports = {}
    versions = struct.unpack('<'+'H'*len(elf.symbols), elf.read(elf.tag(0x6ffffff0), 2*len(elf.symbols))) if elf.tag(0x6ffffff0) else [0]*len(elf.symbols)
    for s, version in zip(elf.symbols, versions):
        if s['shndx'] and s['info'] >> 4 in (1, 2) and s['other'] & 3 in (0, 3):
            if s['name'] not in exports or not version & 0x8000:
                exports[s['name']] = s
    labels = {p[3] for p in elf.loads}
    replacements = {}
    for site in sites:
        address = site['address']
        replacements[address] = (4, f'b Laot_{address:x}')
        labels.add(address+4)
        if site['kind'] == 'x18':
            _, targets = lower_x18(site)
            labels.update(targets)
    if plan:
        for address, target in plan.overrides.items():
            replacements[address] = (4, f'b {target}')
    imports = {}
    for target, kind, index, addend in elf.relocs:
        symbol = elf.symbols[index]
        if kind == 1027:
            value = addend
            labels.add(value)
            replacement = f'.quad Lelf_{value:x}'
        elif kind in (257, 1025, 1026):
            if symbol['shndx']:
                value = ifuncs.get(symbol['value'], symbol['value'])+addend
                labels.add(value)
                replacement = f'.quad Lelf_{value:x}'
            else:
                imports[symbol['name']] = symbol['info'] & 15
                binding = plan.imports.get(index) if plan else '_aot_import_'+symbol['name']
                if binding is None and symbol['info'] >> 4 != 2:
                    raise ValueError(f'unresolved required import {symbol["name"]}')
                replacement = f'.quad {binding or 0}'
        elif kind == 1028:
            replacement = f'.quad {plan.tls_modules.get(index, plan.module_id) if plan else 1}'
        elif kind == 1029:
            replacement = f'.quad {symbol["value"]+addend}'
        elif kind == 1030:
            offset = plan.tls_symbols.get(index, tls_offset+symbol['value']) if plan else tls_offset+symbol['value']
            replacement = f'.quad {offset+addend}'
        elif kind == 1031:
            offset = plan.tls_symbols.get(index, tls_offset+symbol['value']) if plan else tls_offset+symbol['value']
            replacement = f'.quad Laot_tlsdesc\n.quad {offset+addend}'
        elif kind == 1032:
            value = ifuncs[addend]
            labels.add(value)
            replacement = f'.quad Lelf_{value:x}'
        else:
            raise ValueError(f'unsupported AOT relocation {kind}')
        replacements[target] = (16 if kind == 1031 else 8, replacement)
    if plan:
        for index in plan.definitions:
            symbol = elf.symbols[index]
            if symbol['info'] & 15 != 6:
                labels.add(ifuncs.get(symbol['value'], symbol['value']))
    for s in exports.values():
        if s['info'] & 15 != 6:
            labels.add(ifuncs.get(s['value'], s['value']))
    tls = elf.tls
    template = bytearray(tls[6] if tls else 0)
    if tls: template[:tls[5]] = elf.data[tls[2]:tls[2]+tls[5]]
    tls_pointers = []
    # Read RELATIVE relocations within the TLS template (there is also a mapped copy).
    if tls:
        for target, kind, _, addend in elf.relocs:
            if tls[3] <= target < tls[3]+tls[5] and kind == 1027:
                tls_pointers.append((target-tls[3], addend))
                labels.add(addend)
    def array_values(address, size):
        relocated = {target: addend for target, kind, _, addend in elf.relocs if kind == 1027}
        return [relocated.get(address+i, struct.unpack('<Q', elf.read(address+i, 8))[0]) for i in range(0, size, 8)]
    initializers = ([elf.tag(12)] if elf.tag(12) else [])+array_values(elf.tag(25), elf.tag(27))
    finalizers = list(reversed(array_values(elf.tag(26), elf.tag(28))))+([elf.tag(13)] if elf.tag(13) else [])
    labels.update(initializers+finalizers)
    output = []
    filenames = []
    for load_index, (p, segment_name) in enumerate(zip(elf.loads, segment_names)):
        filename = ['rx.bin', 'rw.bin'][load_index] if plan is None else f'load{load_index}.bin'
        filenames.append(filename)
        floor = p[3] & -0x4000
        data = b'\0'*(p[3]-floor)+elf.data[p[2]:p[2]+p[5]]+b'\0'*(p[6]-p[5])
        (directory/filename).write_bytes(data)
        output += [f'.section {segment_name},__image'+(',regular,pure_instructions' if p[1] & 1 else ''), '.p2align 14']
        start, end = floor, p[3]+p[6]
        events = sorted({start, end} | {v for v in labels if start <= v <= end} | {v for v in replacements if start <= v < end})
        cursor = start
        for event in events:
            if event < cursor: continue
            if event > cursor:
                output.append(f'.incbin "{filename}", {cursor-floor}, {event-cursor}')
            output.append(f'Lelf_{event:x}:')
            cursor = event
            if event in replacements:
                size, replacement = replacements[event]
                output.append(replacement)
                cursor += size
    if plan:
        for index, symbol_name in plan.definitions.items():
            symbol = elf.symbols[index]
            if symbol['info'] & 15 == 6:
                continue
            address = ifuncs.get(symbol['value'], symbol['value'])
            output += [f'.globl {symbol_name}', f'.private_extern {symbol_name}', f'.set {symbol_name}, Lelf_{address:x}']
    public_exports = [(s['name'], s) for index, s in enumerate(elf.symbols) if index in plan.public] if plan else list(exports.items())
    for name, s in public_exports:
        if s['info'] & 15 == 6: continue
        address = ifuncs.get(s['value'], s['value'])
        if core and name in PUBLIC_WRAPPERS:
            continue
        output += [f'.globl _g_{name}', f'.set _g_{name}, Lelf_{address:x}']
    for name in ['vsnprintf', 'vfprintf'] if core else []:
        address = exports[name]['value']
        output += [f'.globl _aot_linux_{name}', f'.private_extern _aot_linux_{name}',
                   f'.set _aot_linux_{name}, Lelf_{address:x}']
    output += ['.text', '.p2align 2']
    if any(kind == 1031 for _, kind, _, _ in elf.relocs):
        output += ['Laot_tlsdesc:', 'ldr x0, [x0, #8]', 'ret']
    numbers = set()
    for site in sites:
        address = site['address']
        output += [f'Laot_{address:x}:', *save_frame()]
        if site['kind'] == 'tls':
            output += ['bl _aot_thread_pointer', f'str x0, [sp, #{site["register"]*8}]']
        elif site['kind'] == 'syscall':
            number = site['number']
            if number is None:
                output += ['mov x6, x8', 'bl _aot_dynamic_syscall']
            else:
                numbers.add(number)
                output += [f'bl _aot_nr_{number}']
            output += ['str x0, [sp]']
        else:
            lowered, _ = lower_x18(site)
            output += lowered
            continue
        output += [*restore_frame(), f'b Lelf_{address+4:x}']
    if plan is None or plan.first_tls:
        alignment = plan.tls_alignment if plan else 16
        output += ['.section __DATA,__thread_data,thread_local_regular', f'.p2align {alignment.bit_length()-1}',
                   '.globl _aot_tls_image', '.private_extern _aot_tls_image', '_aot_tls_image:', f'.space {PRE_TCB+16}']
    elif tls:
        output += ['.section __DATA,__thread_data,thread_local_regular', f'.space {plan.tls_padding}']
    tls_labels = {s['value']: [] for s in exports.values() if s['info'] & 15 == 6}
    for name, s in exports.items():
        if s['info'] & 15 == 6: tls_labels[s['value']].append(name)
    tls_replacements = dict(tls_pointers)
    cursor = 0
    for event in sorted(set(tls_labels) | set(tls_replacements) | {len(template)}):
        if event > cursor: output.append('.byte '+','.join(str(b) for b in template[cursor:event]))
        for name in tls_labels.get(event, []): output.append(f'Laot_tls_{name}:')
        cursor = event
        if event in tls_replacements:
            output.append(f'.quad Lelf_{tls_replacements[event]:x}')
            cursor += 8
    if plan is None or plan.first_tls:
        output += ['.section __DATA,__thread_vars,thread_local_variables', '.p2align 3',
                   '.globl _aot_tls_descriptor', '.private_extern _aot_tls_descriptor', '_aot_tls_descriptor:',
                   '.quad __tlv_bootstrap', '.quad 0', '.quad _aot_tls_image']
    for name, s in public_exports:
        if s['info'] & 15 == 6:
            output += [f'.globl _g_{name}', f'_g_{name}:', '.quad __tlv_bootstrap', '.quad 0', f'.quad Laot_tls_{name}']
    output += ['.text']
    init_prefix = '_aot_original_init_' if core else f'_aot_{plan.identifier}_init_' if plan else '_aot_original_init_'
    for i, address in enumerate(initializers):
        output += [f'.globl {init_prefix}{i}', f'.private_extern {init_prefix}{i}',
                   f'.set {init_prefix}{i}, Lelf_{address:x}']
        # These values usually are also local RELATIVE targets, hence already labeled.
    if plan:
        for i, address in enumerate(finalizers):
            name = f'_aot_{plan.identifier}_fini_{i}'
            output += [f'.globl {name}', f'.private_extern {name}', f'.set {name}, Lelf_{address:x}']
    (directory/'image.S').write_text('\n'.join(output)+'\n')
    manifest = {'input': str(Path(source).resolve()), 'exports': len(exports), 'ifuncs': {hex(k): hex(v) for k, v in ifuncs.items()},
                'syscall_sites': [s for s in sites if s['kind'] == 'syscall'], 'tls_sites': sum(s['kind'] == 'tls' for s in sites),
                'x18_sites': x18, 'x18_lowered': True, 'imports': imports, 'numbers': sorted(numbers), 'initializers': initializers,
                'segments': {name: base+(p[3] & -0x4000) for name, p in zip(segment_names, elf.loads)},
                'binary_files': filenames, 'finalizers': finalizers,
                'complete': False}
    (directory/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest
