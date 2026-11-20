"""Which GPU generations can run every CUDA extension in this image?

Build time (Dockerfile): scans every CUDA library with cuobjdump, writes
/workspace/cuda_archs.json and fails the build if a required compute
capability is not covered. Job start (forest_mamba_laz_batch.py) reads that
file to print one GPU info line and to stop early on an unsupported GPU.

A library runs on compute capability X.Y if it holds SASS for X.Z with Z <= Y
(same major), or PTX for any version <= X.Y (JIT compiled by the driver).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

CANDIDATES = ("6.1", "7.0", "7.5", "8.0", "8.6", "8.9", "9.0")
DEFAULT_ROOTS = ("/opt/conda/lib/python3.10/site-packages", "/workspace")
# Libraries shipped by NVIDIA or PyTorch themselves cover all common GPUs.
SKIP = re.compile(r"/(torch/lib|nvidia|triton)/|libcu|libnv|cudnn")
INFO_PATH = Path("/workspace/cuda_archs.json")


def _versions(text: str, prefix: str) -> set[tuple[int, int]]:
    return {(int(m[0]), int(m[1:])) for m in re.findall(rf"{prefix}_(\d\d+)", text)}


def library_archs(path: Path, cuobjdump: str) -> tuple[set, set] | None:
    elf = subprocess.run([cuobjdump, "--list-elf", str(path)], capture_output=True, text=True).stdout
    ptx = subprocess.run([cuobjdump, "--list-ptx", str(path)], capture_output=True, text=True).stdout
    sass, ptx_versions = _versions(elf, "sm"), _versions(ptx, "sm") | _versions(ptx, "compute")
    return (sass, ptx_versions) if sass or ptx_versions else None


def runs_on(sass: set, ptx: set, capability: str) -> bool:
    major, minor = (int(part) for part in capability.split("."))
    return any(m == major and n <= minor for m, n in sass) or any((m, n) <= (major, minor) for m, n in ptx)


def scan(roots: list[str], cuobjdump: str) -> dict:
    libraries = {}
    for root in roots:
        for path in sorted(Path(root).rglob("*.so")):
            if SKIP.search(str(path)) or path.stat().st_size < 100_000:
                continue
            archs = library_archs(path, cuobjdump)
            if archs:
                libraries[str(path)] = archs
    supported = [c for c in CANDIDATES if all(runs_on(s, p, c) for s, p in libraries.values())]
    blockers = {c: sorted("/".join(Path(lib).parts[-2:]) for lib, (s, p) in libraries.items() if not runs_on(s, p, c))
                for c in CANDIDATES if c not in supported}
    return {"supported_compute": supported, "blockers": blockers, "libraries": len(libraries)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require", default="7.5,8.0,8.6", help="compute capabilities that must be covered")
    parser.add_argument("--output", default=str(INFO_PATH))
    parser.add_argument("--cuobjdump", default="/usr/local/cuda/bin/cuobjdump")
    parser.add_argument("roots", nargs="*", default=list(DEFAULT_ROOTS))
    args = parser.parse_args()
    info = scan(args.roots, args.cuobjdump)
    Path(args.output).write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info, indent=1))
    missing = [c for c in args.require.split(",") if c not in info["supported_compute"]]
    if missing:
        sys.exit(f"CUDA libraries do not cover compute {', '.join(missing)}: "
                 + "; ".join(f"{c}: {', '.join(info['blockers'][c])}" for c in missing))


if __name__ == "__main__":
    main()
