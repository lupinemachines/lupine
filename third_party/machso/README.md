# machso

Experimental conversion of ARM64 Linux shared libraries into native macOS dylibs
without their source code.

The compiler discovers ELF dependencies, preserves their machine code, and
generates native replacements for supported Linux syscalls, TLS operations, and
uses of Apple's reserved x18 register. It links the dependency graph into one
Mach-O dylib and signs it on an ARM64 Mac.

Requires [uv](https://docs.astral.sh/uv/), Python 3.10+, Linux dependency binaries,
and an ARM64 Mac with Xcode command-line tools for linking and signing.

```sh
uv sync
uv run python compile_bundle.py library.so build/prepared \
  -L /path/to/linux/libraries --experimental

# On macOS, after transferring the prepared directory:
uv run python link_bundle.py build/prepared build/library.dylib
uv run python sign_bundle.py build/library.dylib
```

Compilation, native linking, and signing are separate file-based stages. Transfer
the prepared directory using your own CI or file transport; machso does not manage
SSH connections. Each stage saves a JSON report. Use `--plan-only` to inspect the
dependency graph without generating code. Add `--include helper.so` for libraries
loaded dynamically; all roots share one dependency closure and Linux runtime.
On Linux, add `--ubuntu-cache /path/to/cache` to acquire missing ARM64 libraries
from Ubuntu Noble's signed package indexes, without a filename-to-package table.
This optional provider requires `gpgv`, `dpkg-deb`, and Ubuntu's archive keyring
(`--ubuntu-keyring` overrides its path); reports record package versions and hashes.

To satisfy an ELF dependency with an existing native dylib, generate its interface
on macOS and pass that file to the portable compiler:

```sh
uv run python inspect_native.py libprovider.dylib provider.json --soname libprovider.so.1
uv run python compile_bundle.py library.so build/prepared -L /path/to/linux/libraries \
  --native-provider provider.json --experimental
uv run python link_bundle.py build/prepared build/library.dylib \
  --native-library libprovider.dylib
```

The linker checks the native install name and required exports before building.
Late symbol lookup includes native dependencies and uses their declared ABI bridges.
Native providers must implement the caller's ABI. Supply `inspect_native.py
--signatures signatures.json` to generate Linux-to-Darwin argument bridges for
fixed scalar functions. The file maps exported names to `arguments` and `result`,
for example `{"foo":{"arguments":["ptr","i32"],"result":"i32"}}`.
Supported types are `i8/u8/i16/u16/i32/u32/i64/u64/ptr/f32/f64`, plus `void` results.
Variadic and aggregate signatures are unsupported; functions without signatures
link directly.

With API headers available, generate signatures automatically from the inspected
exports using the optional `headers` extra (requires Clang):

```sh
uv run --extra headers python inspect_headers.py provider.json typed-provider.json \
  --header /path/to/provider.h -I /path/to/includes
```

Pass `typed-provider.json` to `--native-provider`. It records undeclared exports
and unsupported signatures; using a known unsupported signature fails preparation.
On macOS, use `--sysroot` with ARM64 Linux headers when an API includes Linux
system headers. Parsing always targets the Linux caller's type layout.
Use `--signature-typedef NAME:TYPEDEF` for an older exported API whose signature
is available as a historical function-pointer typedef.

For status-returning APIs that write a function pointer, add
`--function-pointer-output NAME:ARGUMENT:UNAVAILABLE` to header inspection.
`ARGUMENT` is the zero-based pointer-output argument; status zero means success.
Returned declared functions use their ABI bridges. Known unsupported functions
return a null pointer and the supplied unavailable status; native errors retain
their original status and output. Undeclared function pointers pass through.

To make exported scalar functions callable with the native Darwin ABI, inspect
the portable plan against their API headers and supply `--export-interface`:

```sh
uv run python compile_bundle.py library.so build/api.json -L /path/to/linux/libraries --plan-only
uv run --extra headers python inspect_headers.py build/api.json build/typed-api.json --header api.h
uv run python compile_bundle.py library.so build/prepared -L /path/to/linux/libraries \
  --export-interface build/typed-api.json --experimental
```

Internal ELF calls retain the Linux ABI. Declared function-pointer outputs also
translate back to native entries. Unannotated exports retain their original ABI.

This is a prototype with partial Linux ABI support. The glibc platform profile
targets ARM64 glibc 2.39; unsupported syscalls return `ENOSYS`, and some loader
operations remain trap stubs. Successful conversion does not establish that every
exported function works.

```sh
uv run python -m unittest discover -s tests -p 'test_*.py'
```

Some tests require `aarch64-linux-gnu-gcc` or an ARM64 glibc binary.
