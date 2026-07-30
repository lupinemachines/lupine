# GDB Printers

This directory contains a python module that allows pretty-printing a
cuckoohash map.

## Usage

In order to use the libcuckoo pretty printers, add the following to your
`~/.gdbinit`:

```
python
import sys
sys.path.insert(0, '/path/to/libcuckoo/libcuckoo-gdb-printers')
from libcuckoo.printers import register_libcuckoo_printers
register_libcuckoo_printers (None)
end
```
