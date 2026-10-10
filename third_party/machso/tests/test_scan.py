"""Instruction scanning stays bounded and preserves control-flow boundaries."""
import struct
import unittest
from unittest.mock import patch

from aot.emit import Cs, scan


class ScanTests(unittest.TestCase):
    def scan(self, branch):
        code = bytearray(struct.pack('<I', 0xd503201f) * (0x8004 // 4))
        struct.pack_into('<I', code, 0, 0x14001000 if branch else 0xd503201f)
        struct.pack_into('<I', code, 0x3ffc, 0xd2801588)  # mov x8, #172
        struct.pack_into('<I', code, 0x4000, 0xd4000001)  # svc #0
        sizes = []
        def decoder(*args):
            result = Cs(*args)
            disasm = result.disasm
            def bounded(data, address):
                sizes.append(len(data))
                return disasm(data, address)
            result.disasm = bounded
            disasm_lite = result.disasm_lite
            def bounded_lite(data, address):
                sizes.append(len(data))
                return disasm_lite(data, address)
            result.disasm_lite = bounded_lite
            return result
        with patch('aot.emit.executable_sections', return_value=[(0, bytes(code))]), \
                patch('aot.emit.Cs', side_effect=decoder):
            sites, unused = scan(None)
        self.assertTrue(sizes)
        self.assertLessEqual(max(sizes), 0x4000)
        return next(site for site in sites if site['kind'] == 'syscall')

    def test_register_state_crosses_decode_chunks(self):
        self.assertEqual(self.scan(False)['number'], 172)

    def test_branch_into_chunk_resets_register_state(self):
        self.assertIsNone(self.scan(True)['number'])


if __name__ == '__main__':
    unittest.main()
