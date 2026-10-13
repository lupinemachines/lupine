"""Read native API declarations without guessing typedef widths or enum layouts."""
from pathlib import Path
import subprocess
from aot.macho import file_hash


def scalar_type(type_):
    from clang.cindex import TypeKind
    canonical = type_.get_canonical()
    kind = canonical.kind
    if kind == TypeKind.ENUM:
        return scalar_type(canonical.get_declaration().enum_type)
    if kind in (TypeKind.POINTER, TypeKind.CONSTANTARRAY, TypeKind.INCOMPLETEARRAY):
        return 'ptr'
    if kind == TypeKind.VOID:
        return 'void'
    signed = (TypeKind.CHAR_S, TypeKind.SCHAR, TypeKind.SHORT, TypeKind.INT,
              TypeKind.LONG, TypeKind.LONGLONG)
    unsigned = (TypeKind.BOOL, TypeKind.CHAR_U, TypeKind.UCHAR, TypeKind.USHORT,
                TypeKind.UINT, TypeKind.ULONG, TypeKind.ULONGLONG)
    if kind in signed+unsigned and canonical.get_size() in (1, 2, 4, 8):
        return ('i' if kind in signed else 'u')+str(canonical.get_size()*8)
    if kind in (TypeKind.FLOAT, TypeKind.DOUBLE):
        return 'f'+str(canonical.get_size()*8)
    raise ValueError(f'unsupported native type: {type_.spelling}')


def read_signatures(headers, exports, include_dirs=(), clang='clang', language='c', sysroot=None,
                    signature_typedefs=None):
    from clang.cindex import CursorKind, Diagnostic, Index, TypeKind
    resource = subprocess.check_output([clang, '-print-resource-dir'], text=True).strip()
    target = 'aarch64-unknown-linux-gnu'
    args = ['-target', target, '-x', language, '-resource-dir', resource]
    if sysroot:
        args += ['--sysroot', str(Path(sysroot).resolve())]
    for directory in include_dirs:
        args += ['-I', str(Path(directory).resolve())]
    declarations, unsupported = {}, {}
    index = Index.create()
    exports = set(exports)
    signature_typedefs = {} if signature_typedefs is None else signature_typedefs
    if (not isinstance(signature_typedefs, dict) or
            any(name not in exports or not isinstance(type_, str) or not type_
                for name, type_ in signature_typedefs.items())):
        raise ValueError('signature typedefs must map exported names to typedef names')
    typedef_names = set(signature_typedefs.values())
    found_typedefs = set()
    inputs = set()
    for header in headers:
        header = Path(header).resolve()
        if not header.is_file():
            raise ValueError(f'header does not exist: {header}')
        unit = index.parse(str(header), args=args)
        inputs.add(header)
        inputs.update(Path(include.include.name).resolve() for include in unit.get_includes())
        errors = [str(d) for d in unit.diagnostics if d.severity >= Diagnostic.Error]
        if errors:
            raise ValueError('header parsing failed:\n'+'\n'.join(errors))
        # Traverse linkage/namespace declarations, without descending into
        # function bodies or the SDK's extensive documentation comments.
        def visit(cursor):
            for child in cursor.get_children():
                if child.kind in (CursorKind.FUNCTION_DECL, CursorKind.TYPEDEF_DECL):
                    yield child
                elif child.kind in (CursorKind.LINKAGE_SPEC, CursorKind.NAMESPACE,
                                     CursorKind.UNEXPOSED_DECL):
                    yield from visit(child)
        for declaration in visit(unit.cursor):
            if declaration.kind == CursorKind.TYPEDEF_DECL:
                if declaration.spelling not in typedef_names:
                    continue
                found_typedefs.add(declaration.spelling)
                names = [name for name, type_ in signature_typedefs.items() if type_ == declaration.spelling]
                function = declaration.underlying_typedef_type.get_canonical()
                if function.kind == TypeKind.POINTER:
                    function = function.get_pointee().get_canonical()
            else:
                name = declaration.mangled_name or declaration.spelling
                if name not in exports:
                    continue
                names, function = [name], declaration.type
            try:
                if function.kind != TypeKind.FUNCTIONPROTO:
                    raise ValueError('signature typedef must be a fixed function or function pointer')
                if function.is_function_variadic():
                    raise ValueError('variadic native functions are unsupported')
                signature = {'arguments': [scalar_type(p) for p in function.argument_types()],
                             'result': scalar_type(function.get_result())}
            except ValueError as error:
                for name in names:
                    unsupported[name] = str(error)
                    declarations.pop(name, None)
                continue
            for name in names:
                if signature != declarations.get(name, signature):
                    unsupported[name] = 'conflicting native declarations'
                    declarations.pop(name, None)
                elif name not in unsupported:
                    declarations[name] = signature
    missing_typedefs = typedef_names-found_typedefs
    if missing_typedefs:
        raise ValueError(f'signature typedefs not found: {sorted(missing_typedefs)}')
    return {'signatures': declarations, 'unsupported': unsupported,
            'missing_declarations': sorted(exports-set(declarations)-set(unsupported)),
            'target': target, 'signature_typedefs': signature_typedefs,
            'headers': [{'path': str(path), 'sha256': file_hash(path)} for path in sorted(inputs)]}
