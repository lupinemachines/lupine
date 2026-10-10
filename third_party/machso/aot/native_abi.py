"""Bridge fixed scalar AAPCS64 arguments to Darwin's ARM64 calling convention."""

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
    return signatures


def emit_bridge(label, target, signature):
    gp = fp = linux = darwin = 0
    copies, extensions = [], []
    for argument in signature['arguments']:
        size = SIZES[argument]
        floating = argument.startswith('f')
        register = fp if floating else gp
        if floating:
            fp += 1
        else:
            gp += 1
        if register < 8:
            if not floating and size < 4:
                instruction = ('sxt' if argument.startswith('i') else 'uxt')+('b' if size == 1 else 'h')
                extensions.append(f'    {instruction} w{register}, w{register}')
            continue
        darwin = (darwin+size-1)&-size
        copies.append((linux, darwin, size))
        linux += 8
        darwin += size
    saved = (darwin+15)&-16
    frame = saved+16
    lines = ['.text', '.p2align 2', f'.globl {label}', f'.private_extern {label}', f'{label}:',
             f'    sub sp, sp, #{frame}', f'    str x29, [sp, #{saved}]',
             f'    str x30, [sp, #{saved+8}]',
             f'    add x29, sp, #{saved}', f'    add x16, sp, #{frame}', *extensions]
    for source, destination, size in copies:
        load, store, register = {1: ('ldrb', 'strb', 'w17'), 2: ('ldrh', 'strh', 'w17'),
                                 4: ('ldr', 'str', 'w17'), 8: ('ldr', 'str', 'x17')}[size]
        lines += [f'    {load} {register}, [x16, #{source}]',
                  f'    {store} {register}, [sp, #{destination}]']
    lines += [f'    bl {target}', f'    ldr x30, [sp, #{saved+8}]',
              f'    ldr x29, [sp, #{saved}]',
              f'    add sp, sp, #{frame}', '    ret']
    return '\n'.join(lines)+'\n'
