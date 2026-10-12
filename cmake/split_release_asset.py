"""Split the Mac runtime wheel into release assets below GitHub's size limit."""

import hashlib
import sys
from pathlib import Path


def split(source: Path, directory: Path, part_size: int = 1024**3) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    checksums = []
    whole = hashlib.sha256()
    with source.open("rb") as stream:
        number = 0
        while block := stream.read(min(part_size, 1024**2)):
            number += 1
            part = directory / f"{source.name}.part{number:03d}"
            digest = hashlib.sha256()
            size = 0
            with part.open("wb") as output:
                while block:
                    output.write(block)
                    digest.update(block)
                    whole.update(block)
                    size += len(block)
                    if size == part_size:
                        break
                    block = stream.read(min(part_size - size, 1024**2))
            checksums.append(f"{digest.hexdigest()}  {part.name}")
    checksums.append(f"{whole.hexdigest()}  {source.name}")
    (directory / "macos-wheel-SHA256SUMS").write_text("\n".join(checksums) + "\n")


if __name__ == "__main__":
    split(Path(sys.argv[1]), Path(sys.argv[2]))
