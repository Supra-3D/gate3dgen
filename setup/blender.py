"""Install the verified official OCIO-enabled Blender 3.0.1 into the current Conda prefix."""
from __future__ import annotations

import hashlib
import platform
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

NAME = "blender-3.0.1-linux-x64"
URL = f"https://download.blender.org/release/Blender3.0/{NAME}.tar.xz"
SHA256 = "4f17aa3d10ed6e13e6a75479f1a506f58998b8c007812a0886d9254c953e2ae5"


def main():
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "AMD64"}:
        raise RuntimeError("Automatic Blender installation supports Linux x86_64 only")
    from gate3dgen.evaluation.blender_runtime import ensure_runtime
    prefix = Path(sys.prefix)
    link = prefix / "bin/blender"
    if link.exists() or link.is_symlink():
        ensure_runtime(str(link))
        print(f"Using existing validated Blender: {link}")
        return
    destination = prefix / "share/gate3dgen"
    destination.mkdir(parents=True, exist_ok=True)
    binary = destination / NAME / "blender"
    if not binary.exists():
        if binary.parent.exists():
            raise FileExistsError(f"Incomplete installation; inspect before retrying: {binary.parent}")
        with tempfile.TemporaryDirectory(prefix="blender-install-", dir=destination) as temporary:
            archive = Path(temporary) / "blender.tar.xz"
            with urllib.request.urlopen(URL, timeout=120) as response, archive.open("wb") as handle:
                digest = hashlib.sha256()
                while block := response.read(1024 * 1024):
                    digest.update(block)
                    handle.write(block)
            if digest.hexdigest() != SHA256:
                raise ValueError("Official Blender archive SHA256 mismatch")
            with tarfile.open(archive) as bundle:
                bundle.extractall(temporary, filter="data")
            staged = Path(temporary) / NAME
            ensure_runtime(str(staged / "blender"))
            staged.rename(binary.parent)
    ensure_runtime(str(binary))
    link.symlink_to(binary)
    print(f"Installed verified OCIO-enabled Blender: {link}")


if __name__ == "__main__":
    main()
