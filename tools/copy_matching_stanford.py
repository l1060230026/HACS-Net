#!/usr/bin/env python3
import argparse
import os
import re
import shutil
from pathlib import Path
from typing import Set, Dict, List, Tuple


def normalize_name(name: str) -> str:
    """
    Normalize a scene filename by removing a trailing "_S" before the extension.
    Example: "Area_1_conferenceRoom_1_S.npy" -> "Area_1_conferenceRoom_1.npy"
    """
    if name.endswith(".npy"):
        stem = name[:-4]
        if stem.endswith("_S"):
            return f"{stem[:-2]}.npy"
        return name
    # If no extension provided, still strip trailing _S
    return re.sub(r"_S$", "", name)


def collect_names(directory: Path) -> Set[str]:
    return {p.name for p in directory.glob("*.npy")}


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Copy .npy files from stanford_indoor3d to stanford_small if their "
            "normalized names (strip trailing _S) appear in bim_scan."
        )
    )
    parser.add_argument(
        "--bim-scan",
        type=Path,
        default=Path("data/bim_scan"),
        help="Directory containing BIM scan .npy files (source of names)",
    )
    parser.add_argument(
        "--stanford-src",
        type=Path,
        default=Path("data/stanford_indoor3d"),
        help="Directory containing stanford_indoor3d .npy files (copy source)",
    )
    parser.add_argument(
        "--stanford-dst",
        type=Path,
        default=Path("data/stanford_small"),
        help="Destination directory to write matched stanford files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print actions without copying files",
    )
    args = parser.parse_args()

    bim_dir: Path = args.bim_scan
    stanford_src: Path = args.stanford_src
    stanford_dst: Path = args.stanford_dst

    if not bim_dir.is_dir():
        raise SystemExit(f"bim_scan directory not found: {bim_dir}")
    if not stanford_src.is_dir():
        raise SystemExit(f"stanford_indoor3d directory not found: {stanford_src}")
    stanford_dst.mkdir(parents=True, exist_ok=True)

    bim_files = collect_names(bim_dir)
    stanford_files = collect_names(stanford_src)

    # Build lookup for stanford by normalized name -> original filename
    normalized_to_src: Dict[str, str] = {}
    for fname in stanford_files:
        norm = normalize_name(fname)
        # In rare duplicates after normalization, prefer first occurrence
        normalized_to_src.setdefault(norm, fname)

    to_copy: List[Tuple[Path, Path]] = []
    missing: List[str] = []

    for bim_name in bim_files:
        norm = normalize_name(bim_name)
        src_name = normalized_to_src.get(norm)
        if src_name is None:
            missing.append(bim_name)
            continue
        src_path = stanford_src / src_name
        dst_path = stanford_dst / src_name
        to_copy.append((src_path, dst_path))

    # Execute copies
    copied_count = 0
    for src_path, dst_path in to_copy:
        if args.dry_run:
            print(f"DRY RUN: copy {src_path} -> {dst_path}")
            continue
        shutil.copy2(src_path, dst_path)
        copied_count += 1

    # Report
    print(f"Matched {len(to_copy)} files. Copied: {copied_count}.")
    if missing:
        print(f"Warning: {len(missing)} names in bim_scan had no match in stanford_indoor3d.")
        # Show a few examples to help debugging
        for name in sorted(missing)[:20]:
            print(f"  missing -> {name} (normalized: {normalize_name(name)})")


if __name__ == "__main__":
    main()


