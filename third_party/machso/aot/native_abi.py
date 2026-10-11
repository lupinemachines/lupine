"""Bridge fixed scalar AAPCS64 arguments to Darwin's ARM64 calling convention."""
import json

SIZES = {'i8': 1, 'u8': 1, 'i16': 2, 'u16': 2, 'i32': 4, 'u32': 4,
         'i64': 8, 'u64': 8, 'ptr': 8, 'f32': 4, 'f64': 8}


def validate_signatures(provider):
    signatures = provider.get('signatures', {})
    if not isinstance(signatures, dict):
        raise ValueError('native signatures must be a mapping')
    for name, signature in signatures.items():
        if name not in provider['exports'] or not isinstance(signature, dict):
            raise ValueError(f'{name}: signature must describe an exported function')
        arguments = signature.get('arguments')
        if (not isinstance(arguments, list) or len(arguments) > 256 or
                any(not isinstance(arg, str) or arg not in SIZES for arg in arguments) or
                signature.get('result') not in (*SIZES, 'void') or signature.get('variadic')):
            raise ValueError(f'{name}: only fixed scalar native signatures are supported')
        output = signature.get('function_pointer_output')
        if output is not None:
            index = output.get('argument') if isinstance(output, dict) else None
            if (type(index) is not int or not 0 <= index < len(arguments) or
                    arguments[index] != 'ptr' or signature['result'] not in ('i32', 'u32') or
                    type(output.get('unavailable_result')) is not int or
                    not 0 < output['unavailable_result'] < 2**31):
                raise ValueError(f'{name}: function pointer output requires a pointer argument, '
                                 '32-bit status result, and positive unavailable status')
    return signatures


def emit_bridge(label, target, signature, pointer_resolver=None, caller_abi='linux', indirect=False, private=True):
    if caller_abi not in ('linux', 'darwin'):
        raise ValueError('unsupported bridge caller ABI')
    gp = fp = linux = darwin = 0
    copies, extensions = [], []
    output = signature.get('function_pointer_output')
    output_location = None
    for index, argument in enumerate(signature['arguments']):
        size = SIZES[argument]
        floating = argument.startswith('f')
        register = fp if floating else gp
        if floating:
            fp += 1
        else:
            gp += 1
        if register < 8:
            if output and index == output['argument']:
                output_location = f'    mov x19, x{register}'
            if not floating and size < 4:
                instruction = ('sxt' if argument.startswith('i') else 'uxt')+('b' if size == 1 else 'h')
                extensions.append(f'    {instruction} w{register}, w{register}')
            continue
        darwin = (darwin+size-1)&-size
        if output and index == output['argument']:
            offset = linux if caller_abi == 'linux' else darwin
            output_location = f'    ldr x19, [x16, #{offset}]'
        copies.append((linux, darwin, size) if caller_abi == 'linux' else (darwin, linux, size))
        linux += 8
        darwin += size
    stack = darwin if caller_abi == 'linux' else linux
    saved = (stack+15)&-16
    frame = saved+(32 if output else 16)
    lines = ['.text', '.p2align 2', f'.globl {label}',
             *([f'.private_extern {label}'] if private else []), f'{label}:',
             f'    sub sp, sp, #{frame}', f'    str x29, [sp, #{saved}]',
             f'    str x30, [sp, #{saved+8}]',
             f'    add x29, sp, #{saved}', f'    add x16, sp, #{frame}', *extensions]
    if output:
        if not pointer_resolver:
            raise ValueError('function pointer output requires its provider resolver')
        lines += [f'    str x19, [sp, #{saved+16}]', f'    str x20, [sp, #{saved+24}]', output_location]
    for source, destination, size in copies:
        load, store, register = {1: ('ldrb', 'strb', 'w17'), 2: ('ldrh', 'strh', 'w17'),
                                 4: ('ldr', 'str', 'w17'), 8: ('ldr', 'str', 'x17')}[size]
        lines += [f'    {load} {register}, [x16, #{source}]',
                  f'    {store} {register}, [sp, #{destination}]']
    if indirect:
        # ELF images can sit beyond ADRP/BL reach of the native prefix.
        lines += [f'    adrp x16, {target}@GOTPAGE', f'    ldr x16, [x16, {target}@GOTPAGEOFF]',
                  '    blr x16']
    else:
        lines += [f'    bl {target}']
    if output:
        # Only successful queries own the output slot. Preserve native error
        # statuses and leave their output untouched, even if it is uninitialized.
        done = label+'_pointer_done'
        returned = label+'_pointer_return'
        unavailable = label+'_pointer_unavailable'
        status = output['unavailable_result']
        lines += [f'    cbnz w0, {done}', '    mov x20, x0',
                  f'    cbz x19, {unavailable}', '    ldr x0, [x19]',
                  f'    bl {pointer_resolver}', '    str x0, [x19]',
                  f'    cbnz x0, {returned}',
                  f'{unavailable}:', f'    movz w20, #{status & 0xffff}',
                  f'    movk w20, #{status >> 16}, lsl #16',
                  f'{returned}:', '    mov x0, x20', f'{done}:',
                  f'    ldr x19, [sp, #{saved+16}]', f'    ldr x20, [sp, #{saved+24}]']
    lines += [f'    ldr x30, [sp, #{saved+8}]',
              f'    ldr x29, [sp, #{saved}]',
              f'    add sp, sp, #{frame}', '    ret']
    return '\n'.join(lines)+'\n'


def emit_pointer_resolver(provider):
    """Translate only declared native function addresses; unknown pointers pass through."""
    declarations, rows = [], []
    signatures = provider.native.get('signatures', {})
    unsupported = provider.native.get('signature_report', {}).get('unsupported', {})
    definitions = sorted(enumerate(provider.elf.symbols), key=lambda item: item[1]['name'] not in unsupported)
    for index, symbol in definitions:
        name = symbol['name']
        if name not in signatures and name not in unsupported:
            continue
        raw = f'aot_raw_{provider.identifier}_{index}'
        declarations.append(f'extern const unsigned char {raw}[] __asm__({json.dumps("_"+name)});')
        address = '0'
        if name in signatures:
            address = f'aot_callable_{provider.identifier}_{index}'
            target = f'_aot_native_{provider.identifier}_s{index}'
            declarations.append(f'extern const unsigned char {address}[] __asm__({json.dumps(target)});')
        rows.append('{'+raw+','+address+'}')
    function = f'aot_native_{provider.identifier}_function_pointer'
    return ('\n'.join(declarations)+'\n'
            f'__attribute__((visibility("hidden"))) void *{function}(void *address) {{\n'
            '    static const struct {const void *native,*callable;} entries[]={'+','.join(rows)+'};\n'
            '    for (unsigned i=0;i<sizeof(entries)/sizeof(entries[0]);i++)\n'
            '        if (address==entries[i].native) return (void *)entries[i].callable;\n'
            '    return address;\n}\n')


def emit_export_pointer_resolver(exports, native_bridges, unsupported=()):
    """Return native-callable entries for pointers escaping the Linux ABI domain."""
    declarations, rows = [], []
    pairs = [(node.symbol_name(index), None) for node, index in unsupported]
    pairs += [(node.symbol_name(index), '_g_'+name)
             for name, (node, index, _) in exports.items()]
    pairs += [(label, '_'+name) for label, (name, _) in native_bridges.items()]
    for index, (source, target) in enumerate(pairs):
        for side, symbol in (('source', source), ('target', target)):
            if symbol is not None:
                declarations.append(f'extern const unsigned char aot_export_{side}_{index}[] __asm__({json.dumps(symbol)});')
        address = f'aot_export_target_{index}' if target is not None else '0'
        rows.append(f'{{aot_export_source_{index},{address}}}')
    return ('\n'.join(declarations)+'\n'
            'static const struct {const void *source,*target;} aot_export_entries[]={'+','.join(rows)+'};\n'
            '__attribute__((visibility("hidden"))) void *aot_export_function_pointer(void *address) {\n'
            '    for (unsigned i=0;i<sizeof(aot_export_entries)/sizeof(aot_export_entries[0]);i++)\n'
            '        if (address==aot_export_entries[i].source) return (void *)aot_export_entries[i].target;\n'
            '    return address;\n}\n'
            '__attribute__((visibility("hidden"))) void *aot_import_function_pointer(void *address) {\n'
            '    for (unsigned i=0;i<sizeof(aot_export_entries)/sizeof(aot_export_entries[0]);i++)\n'
            '        if (aot_export_entries[i].target && address==aot_export_entries[i].target)\n'
            '            return (void *)aot_export_entries[i].source;\n'
            '    return address;\n}\n')


def emit_export_metadata(names):
    names = sorted(set(names))
    return ('#include <string.h>\n'
            'int g_machso_native_export_abi(const char *name) {\n'
            '    static const char *const names[]={'+','.join(json.dumps(name) for name in names)+'};\n'
            '    if (!name) return 0;\n'
            '    unsigned low=0,high=sizeof(names)/sizeof(names[0]);\n'
            '    while (low<high) {\n'
            '        unsigned middle=low+(high-low)/2;\n'
            '        int order=strcmp(names[middle],name);\n'
            '        if (!order) return 1;\n'
            '        if (order<0) low=middle+1; else high=middle;\n'
            '    }\n    return 0;\n}\n')
