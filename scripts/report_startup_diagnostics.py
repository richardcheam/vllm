#!/usr/bin/env python3
"""Report startup cache locations and derived artifacts without modifying them."""

from __future__ import annotations

import argparse
import json
import os
import stat
from pathlib import Path
from typing import Any


def _cache_root() -> tuple[str | None, Path]:
    configured = os.environ.get("VLLM_CACHE_ROOT")
    resolved = (
        Path(configured).expanduser()
        if configured
        else Path.home() / ".cache" / "vllm"
    )
    return configured, resolved


def _size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return total


def _artifact_summary(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        return {"exists": False, "entries": 0, "files": 0}
    entries = files = 0
    for _, dirs, names in os.walk(path):
        entries += len(dirs) + len(names)
        files += len(names)
    return {"exists": True, "entries": entries, "files": files}


def _mount_type(path: Path) -> str | None:
    try:
        mounts = Path("/proc/mounts").read_text().splitlines()
    except OSError:
        return None
    best_mount = ""
    best_type = None
    path_text = str(path.resolve())
    for line in mounts:
        fields = line.split()
        if len(fields) < 3:
            continue
        mount_point = fields[1].replace("\\040", " ")
        if path_text == mount_point or path_text.startswith(mount_point.rstrip("/") + "/"):
            if len(mount_point) > len(best_mount):
                best_mount = mount_point
                best_type = fields[2]
    return best_type


def _describe_path(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return result
    try:
        info = path.stat()
        result.update(
            {
                "mode": stat.filemode(info.st_mode),
                "uid": info.st_uid,
                "gid": info.st_gid,
                "readable": os.access(path, os.R_OK),
                "writable": os.access(path, os.W_OK),
                "size_bytes": _size_bytes(path),
                "filesystem": _mount_type(path),
            }
        )
    except OSError as exc:
        result["error"] = str(exc)
    return result


def collect_report() -> dict[str, Any]:
    configured, root = _cache_root()
    compile_cache = root / "torch_compile_cache"
    modelinfos = root / "modelinfos"
    startup_plan = root / "startup_plan"
    return {
        "environment": {
            "HOME": os.environ.get("HOME"),
            "VLLM_CACHE_ROOT": configured,
            "resolved_cache_root": str(root),
            "expanded_configured_root": (
                str(Path(configured).expanduser()) if configured else None
            ),
            "path_consistent": configured is None
            or Path(configured).expanduser().resolve() == root.resolve(),
        },
        "paths": {
            "cache_root": _describe_path(root),
            "torch_compile_cache": _describe_path(compile_cache),
            "modelinfos": _describe_path(modelinfos),
            "startup_plan": _describe_path(startup_plan),
        },
        "artifacts": {
            "torch_compile_cache": _artifact_summary(compile_cache),
            "startup_plan": _artifact_summary(startup_plan),
            "aot_candidates": {
                suffix: _artifact_summary(root / suffix)
                for suffix in ("aot_compile", "aot_cache", "inductor_cache")
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args()
    report = collect_report()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    for section, values in report.items():
        print(f"[{section}]")
        print(json.dumps(values, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
