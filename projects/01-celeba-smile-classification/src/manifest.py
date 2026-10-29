"""Validate image manifests without requiring the ML dependencies."""

import csv
from pathlib import Path


def read_manifest(path):
    """Read a nonempty CSV with unique image IDs and binary integer labels."""
    rows = []
    seen = set()
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"id", "label"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: expected id and label columns")
        for line, row in enumerate(reader, 2):
            image_id = (row.get("id") or "").strip()
            label = (row.get("label") or "").strip()
            if not image_id or Path(image_id).name != image_id or image_id in {".", ".."}:
                raise ValueError(f"{path}:{line}: id must be an image filename")
            if label not in {"0", "1"}:
                raise ValueError(f"{path}:{line}: label must be 0 or 1")
            if image_id in seen:
                raise ValueError(f"{path}:{line}: duplicate image ID {image_id}")
            seen.add(image_id)
            rows.append((image_id, int(label)))
    if not rows:
        raise ValueError(f"{path}: manifest is empty")
    return rows


def validate_splits(train_rows, valid_rows):
    """Prevent the same image from appearing in both evaluation splits."""
    overlap = {row[0] for row in train_rows} & {row[0] for row in valid_rows}
    if overlap:
        raise ValueError(f"Training and validation share {len(overlap)} image IDs")
