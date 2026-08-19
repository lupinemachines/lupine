#ifndef LUPINE_ELF_H
#define LUPINE_ELF_H

#if __has_include(<elf.h>)
#include <elf.h>
#else

// CUDA cubins are ELF images even when the client itself is built for a host
// platform that does not provide <elf.h> (notably macOS). Keep the small
// subset of the ELF64 ABI used by the module-image parser local and portable.
#include <cstdint>

using Elf64_Addr = std::uint64_t;
using Elf64_Off = std::uint64_t;
using Elf64_Half = std::uint16_t;
using Elf64_Word = std::uint32_t;
using Elf64_Xword = std::uint64_t;
using Elf64_Sxword = std::int64_t;

constexpr int EI_NIDENT = 16;

struct Elf64_Ehdr {
  unsigned char e_ident[EI_NIDENT];
  Elf64_Half e_type;
  Elf64_Half e_machine;
  Elf64_Word e_version;
  Elf64_Addr e_entry;
  Elf64_Off e_phoff;
  Elf64_Off e_shoff;
  Elf64_Word e_flags;
  Elf64_Half e_ehsize;
  Elf64_Half e_phentsize;
  Elf64_Half e_phnum;
  Elf64_Half e_shentsize;
  Elf64_Half e_shnum;
  Elf64_Half e_shstrndx;
};

struct Elf64_Phdr {
  Elf64_Word p_type;
  Elf64_Word p_flags;
  Elf64_Off p_offset;
  Elf64_Addr p_vaddr;
  Elf64_Addr p_paddr;
  Elf64_Xword p_filesz;
  Elf64_Xword p_memsz;
  Elf64_Xword p_align;
};

struct Elf64_Shdr {
  Elf64_Word sh_name;
  Elf64_Word sh_type;
  Elf64_Xword sh_flags;
  Elf64_Addr sh_addr;
  Elf64_Off sh_offset;
  Elf64_Xword sh_size;
  Elf64_Word sh_link;
  Elf64_Word sh_info;
  Elf64_Xword sh_addralign;
  Elf64_Xword sh_entsize;
};

struct Elf64_Sym {
  Elf64_Word st_name;
  unsigned char st_info;
  unsigned char st_other;
  Elf64_Half st_shndx;
  Elf64_Addr st_value;
  Elf64_Xword st_size;
};

#define ELFMAG "\177ELF"
#define SELFMAG 4
#define EI_CLASS 4
#define ELFCLASS64 2
#define SHT_SYMTAB 2
#define SHT_NOBITS 8
#define SHT_DYNSYM 11
#define STT_FUNC 2
#define ELF64_ST_TYPE(info) ((info) & 0x0f)

#endif

#endif
