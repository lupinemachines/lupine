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
dependency graph without generating code.

This is a prototype with partial Linux ABI support. The glibc platform profile
targets ARM64 glibc 2.39; unsupported syscalls return `ENOSYS`, and some loader
operations remain trap stubs. Successful conversion does not establish that every
exported function works.

```sh
uv run python -m unittest discover -s tests -p 'test_*.py'
```

Some tests require `aarch64-linux-gnu-gcc` or an ARM64 glibc binary.
