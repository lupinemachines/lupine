"""Inlining tiny binary spans preserves their bytes and zero-filled boundaries."""
import struct
from types import SimpleNamespace
import unittest

from aot.emit import literal_bytes, load_bytes


class LiteralBytesTests(unittest.TestCase):
    def test_words_and_partial_tails_preserve_every_byte(self):
        for size in range(257):
            data = bytes((i*79+size) % 256 for i in range(size))
            result = bytearray()
            for line in literal_bytes(data):
                directive, values = line.split(' ', 1)
                for value in values.split(','):
                    number = int(value, 0)
                    result.extend(struct.pack('<Q', number) if directive=='.quad' else bytes([number]))
            self.assertEqual(result, data, size)

    def test_load_span_preserves_padding_file_data_and_bss(self):
        elf = SimpleNamespace(data=b'headerABCDtrailer')
        # Offset 6 contains ABCD; the LOAD starts at address 10 and has 4 BSS bytes.
        load = (1, 6, 6, 10, 0, 4, 8, 16)
        self.assertEqual(load_bytes(elf, load, 8, 10), b'\0\0ABCD'+b'\0'*4)
        self.assertEqual(load_bytes(elf, load, 11, 5), b'BCD\0\0')
        self.assertEqual(load_bytes(elf, load, 15, 3), b'\0'*3)
