"""
Dataset Source Loader Module
Discovers, verifies, normalizes, and deduplicates images from heterogeneous sources:
- PlantVillage
- PlantDoc
- PlantWild
- Real-world Field / Agronomist datasets
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image

def compute_md5(file_path: Path, chunk_size: int = 65536) -> str:
    """Computes exact MD5 checksum of an image file."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def compute_dhash(file_path: Path, hash_size: int = 8) -> Optional[str]:
    """
    Computes perceptual difference hash (dHash) to detect near-duplicate images.
    Returns 64-bit hexadecimal string or None if unreadable.
    """
    try:
        with Image.open(file_path) as img:
            img = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
            pixels = list(img.getdata())
            diff = []
            for row in range(hash_size):
                for col in range(hash_size):
                    left = pixels[row * (hash_size + 1) + col]
                    right = pixels[row * (hash_size + 1) + col + 1]
                    diff.append(left > right)
            decimal_value = 0
            hex_string = []
            for index, value in enumerate(diff):
                if value:
                    decimal_value += 2 ** (index % 8)
                if (index % 8) == 7:
                    hex_string.append(hex(decimal_value)[2:].rjust(2, "0"))
                    decimal_value = 0
            return "".join(hex_string)
    except Exception:
        return None

def verify_image_integrity(file_path: Path, min_dim: int = 32) -> Tuple[bool, Optional[str], Optional[int], Optional[int], Optional[int]]:
    """
    Validates image file readability, dimensions, and channel integrity.
    Returns (is_valid, error_reason, width, height, channels).
    """
    try:
        if not file_path.exists() or file_path.stat().st_size == 0:
            return False, "Empty or non-existent file", None, None, None

        with Image.open(file_path) as img:
            img.verify()

        with Image.open(file_path) as img:
            width, height = img.size
            channels = len(img.getbands())

            if width < min_dim or height < min_dim:
                return False, f"Image dimensions ({width}x{height}) below minimum threshold ({min_dim}x{min_dim})", width, height, channels

            return True, None, width, height, channels
    except Exception as e:
        return False, f"Corrupted image: {str(e)}", None, None, None

class LabelNormalizer:
    """
    Normalizes arbitrary folder/source label names into canonical 'crop__disease' labels.
    Never silently guesses unknown disease labels.
    """
    def __init__(self, catalog_path: Optional[Path] = None, sources_config_path: Optional[Path] = None):
        base_dir = Path(__file__).resolve().parent
        catalog_path = catalog_path or (base_dir / "crop_catalog.json")
        sources_config_path = sources_config_path or (base_dir / "dataset_sources.json")

        self.valid_canonical_labels = set()
        self.crop_catalog = {}
        if catalog_path.exists():
            with open(catalog_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for crop in data.get("crops", []):
                    c_id = crop["crop_id"]
                    self.crop_catalog[c_id] = crop
                    for d in crop.get("diseases", []):
                        self.valid_canonical_labels.add(d["canonical_label"])

        self.source_overrides = {}
        if sources_config_path.exists():
            with open(sources_config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                for s in cfg.get("dataset_sources", []):
                    self.source_overrides[s["source_id"]] = s.get("label_mapping_overrides", {})

    def normalize(self, raw_label: str, source_id: str) -> Tuple[Optional[str], Optional[str], Optional[str], str]:
        """
        Resolves raw label to (crop, disease, canonical_label, status).
        Status is 'canonical', 'mapped_override', or 'unmapped'.
        """
        raw_clean = raw_label.strip()

        # Check direct canonical match
        if raw_clean in self.valid_canonical_labels:
            crop, disease = raw_clean.split("__", 1)
            return crop, disease, raw_clean, "canonical"

        # Check source-specific override mapping
        overrides = self.source_overrides.get(source_id, {})
        if raw_clean in overrides:
            target = overrides[raw_clean]
            if target in self.valid_canonical_labels:
                crop, disease = target.split("__", 1)
                return crop, disease, target, "mapped_override"

        # Check standardized pattern 'crop___disease' or 'crop__disease' (case-insensitive)
        cleaned = raw_clean.replace("___", "__").replace(" ", "_").lower()
        if cleaned in self.valid_canonical_labels:
            crop, disease = cleaned.split("__", 1)
            return crop, disease, cleaned, "canonical"

        # Cannot map with certainty: route to review/unmapped
        return None, None, "review/unmapped", "unmapped"

class DatasetScanner:
    """
    Scans directories, discovers image files, tracks sources, detects duplicates,
    and returns rich image discovery records.
    """
    def __init__(self, catalog_path: Optional[Path] = None, sources_path: Optional[Path] = None):
        self.normalizer = LabelNormalizer(catalog_path, sources_path)
        self.seen_hashes: Dict[str, str] = {}  # hash -> first image_id

    def scan_directory(self, root_path: Path, source_name: str) -> List[Dict[str, Any]]:
        """
        Scans a directory where subdirectories represent raw label classes.
        Returns list of structured image record dictionaries.
        """
        records = []
        if not root_path.exists():
            return records

        supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        for item in os.scandir(root_path):
            if item.is_dir():
                raw_label = item.name
                crop, disease, canonical, status = self.normalizer.normalize(raw_label, source_name)

                for entry in os.scandir(item.path):
                    if entry.is_file() and Path(entry.name).suffix.lower() in supported_exts:
                        file_path = Path(entry.path)
                        is_valid, err, width, height, channels = verify_image_integrity(file_path)
                        
                        file_hash = compute_md5(file_path) if is_valid else ""
                        d_hash = compute_dhash(file_path) if is_valid else ""
                        
                        image_id = f"{source_name}_{file_hash[:12]}" if file_hash else f"{source_name}_corrupted_{entry.name}"
                        
                        is_duplicate = False
                        duplicate_of = None
                        if file_hash:
                            if file_hash in self.seen_hashes:
                                is_duplicate = True
                                duplicate_of = self.seen_hashes[file_hash]
                            else:
                                self.seen_hashes[file_hash] = image_id

                        record = {
                            "image_id": image_id,
                            "source": source_name,
                            "path": str(file_path.resolve()),
                            "crop": crop,
                            "disease": disease,
                            "canonical_label": canonical,
                            "status": status,
                            "is_valid": is_valid,
                            "error": err or "",
                            "width": width or 0,
                            "height": height or 0,
                            "channels": channels or 0,
                            "md5_hash": file_hash,
                            "dhash": d_hash or "",
                            "is_duplicate": is_duplicate,
                            "duplicate_of": duplicate_of or ""
                        }
                        records.append(record)

        return records

