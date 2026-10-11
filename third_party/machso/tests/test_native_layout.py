"""Native C++ lookup tables must not overlap the fixed ELF segments."""
import struct
import tempfile
import unittest
from pathlib import Path
from machso import segment, section
from link_bundle import reserve_native_space


class NativeLayoutTests(unittest.TestCase):
    def test_native_sections_shift_elf_images_without_counting_custom_blobs(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            sections=[section('__cstring','__TEXT',0,5*1024**2,0,0),
                      section('__image','__N0L0',0,1024**3,0,0)]
            command=segment('',0,0,0,7,sections)
            header=struct.pack('<IIIIIIII',0xfeedfacf,0x100000c,0,1,1,len(command),0,0)
            (directory/'native.o').write_bytes(header+command)
            report={'segments':{'__N0T':0x400000,'__N0L0':0x404000},
                    'components':{'root.so':{'segments':{'__N0T':0x400000,'__N0L0':0x404000}}},
                    'native_build':{'linker_flags':['-segaddr','__N0T','0x400000',
                                                  '-segaddr','__N0L0','0x404000']}}
            reserve_native_space(report,directory,['native.o'])
            self.assertEqual(report['native_build']['native_space_shift'],2*1024**2)
            self.assertEqual(report['segments']['__N0T'],6*1024**2)
            self.assertEqual(report['components']['root.so']['segments'],report['segments'])
            self.assertEqual(report['segments']['__N0L0']-report['segments']['__N0T'],0x4000)
            self.assertEqual(report['native_build']['linker_flags'][2],'0x600000')
