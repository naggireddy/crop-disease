"""
Dataset Merge & Quality Audit Engine
Discovers images from PlantVillage, PlantDoc, PlantWild, and Field data,
resolves canonical labels, detects duplicates and corruption, and compiles
dataset_manifest.csv alongside dataset_quality_check.md.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

# Add parent directory to sys.path for internal imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from source_loaders import DatasetScanner

def run_merge(
    sources_config_path: Path,
    output_manifest_path: Path,
    output_report_path: Path,
    additional_roots: Dict[str, str] = None
) -> pd.DataFrame:
    """Scans all configured dataset sources and compiles a unified manifest and quality report."""
    base_dir = Path(__file__).resolve().parent
    catalog_path = base_dir / "crop_catalog.json"
    scanner = DatasetScanner(catalog_path=catalog_path, sources_path=sources_config_path)

    with open(sources_config_path, "r", encoding="utf-8") as f:
        sources_cfg = json.load(f)

    all_records: List[Dict[str, Any]] = []
    source_stats: Dict[str, Dict[str, int]] = {}

    for src in sources_cfg.get("dataset_sources", []):
        s_id = src["source_id"]
        env_var = src.get("env_var", "")
        default_rel = src.get("default_relative_path", "")

        # Look up path: CLI override > Env variable > Default relative path
        configured_path_str = None
        if additional_roots and s_id in additional_roots and additional_roots[s_id]:
            configured_path_str = additional_roots[s_id]
        elif env_var and os.getenv(env_var):
            configured_path_str = os.getenv(env_var)
        elif default_rel:
            repo_root = base_dir.parent
            candidate = repo_root / default_rel
            if candidate.exists():
                configured_path_str = str(candidate)

        if configured_path_str:
            target_path = Path(configured_path_str)
            if target_path.exists():
                print(f"[*] Scanning source '{s_id}' at: {target_path}")
                recs = scanner.scan_directory(target_path, s_id)
                all_records.extend(recs)
                source_stats[s_id] = {
                    "total_scanned": len(recs),
                    "valid": sum(1 for r in recs if r["is_valid"] and not r["is_duplicate"]),
                    "corrupted": sum(1 for r in recs if not r["is_valid"]),
                    "duplicates": sum(1 for r in recs if r["is_duplicate"]),
                    "unmapped": sum(1 for r in recs if r["status"] == "unmapped")
                }
            else:
                print(f"[-] Configured directory for '{s_id}' not found on disk: {target_path}")
                source_stats[s_id] = {"status": "not_mounted_or_missing", "path": str(target_path)}
        else:
            source_stats[s_id] = {"status": "not_configured"}

    if not all_records:
        print("[!] No images discovered from configured paths. Generating empty/template manifest structure.")
        columns = [
            "image_id", "source", "path", "crop", "disease", "canonical_label",
            "status", "is_valid", "error", "width", "height", "channels",
            "md5_hash", "dhash", "is_duplicate", "duplicate_of", "split"
        ]
        df = pd.DataFrame(columns=columns)
        df.to_csv(output_manifest_path, index=False)
        generate_quality_report(df, source_stats, output_report_path)
        return df

    df = pd.DataFrame(all_records)
    if "split" not in df.columns:
        df["split"] = "unassigned"

    # Save compiled manifest
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_manifest_path, index=False)
    print(f"[+] Dataset manifest successfully written to: {output_manifest_path} ({len(df)} records)")

    # Generate Markdown Quality Report
    generate_quality_report(df, source_stats, output_report_path)
    return df

def generate_quality_report(df: pd.DataFrame, source_stats: Dict[str, Any], report_path: Path):
    """Generates detailed Markdown report on dataset health, distribution, and data integrity."""
    total_records = len(df)
    valid_count = len(df[df["is_valid"] & (~df["is_duplicate"])]) if total_records > 0 else 0
    corrupt_count = len(df[~df["is_valid"]]) if total_records > 0 else 0
    duplicate_count = len(df[df["is_duplicate"]]) if total_records > 0 else 0
    unmapped_count = len(df[df["status"] == "unmapped"]) if total_records > 0 else 0

    lines = [
        "# Dataset Quality and Ingestion Audit Report",
        "",
        "## 1. Executive Summary",
        "",
        f"- **Total Image Records Discovered:** {total_records}",
        f"- **Verified Valid & Unique Images:** {valid_count}",
        f"- **Corrupted / Truncated Images:** {corrupt_count}",
        f"- **Identical / Near-Duplicate Images:** {duplicate_count}",
        f"- **Unmapped / Unknown Disease Labels:** {unmapped_count} *(Routed to review/unmapped, never silently guessed)*",
        "",
        "## 2. Ingestion Source Statistics",
        "",
        "| Source ID | Scanned Files | Valid & Unique | Corrupted | Duplicates | Unmapped | Status |",
        "|-----------|---------------|----------------|-----------|------------|----------|--------|"
    ]

    for s_id, stats in source_stats.items():
        if "total_scanned" in stats:
            lines.append(
                f"| `{s_id}` | {stats['total_scanned']} | {stats['valid']} | "
                f"{stats['corrupted']} | {stats['duplicates']} | {stats['unmapped']} | Configured & Scanned |"
            )
        else:
            lines.append(f"| `{s_id}` | 0 | 0 | 0 | 0 | 0 | {stats.get('status', 'Not found')} |")

    lines.extend([
        "",
        "## 3. Class Distribution & Balance",
        ""
    ])

    if total_records > 0 and valid_count > 0:
        valid_df = df[df["is_valid"] & (~df["is_duplicate"]) & (df["status"] != "unmapped")]
        
        # Per crop distribution
        crop_counts = valid_df["crop"].value_counts()
        lines.append("### Images Per Crop (20-Crop Taxonomy)")
        lines.append("")
        lines.append("| Crop | Unique Count | Percentage |")
        lines.append("|------|--------------|------------|")
        for crop, count in crop_counts.items():
            pct = (count / valid_count) * 100
            lines.append(f"| `{crop}` | {count} | {pct:.1f}% |")

        # Healthy vs Diseased
        lines.append("")
        lines.append("### Healthy vs Diseased Ratio")
        healthy_count = len(valid_df[valid_df["disease"] == "healthy"])
        diseased_count = len(valid_df[valid_df["disease"] != "healthy"])
        lines.append(f"- **Healthy Samples:** {healthy_count} ({(healthy_count / max(1, valid_count)) * 100:.1f}%)")
        lines.append(f"- **Diseased Samples:** {diseased_count} ({(diseased_count / max(1, valid_count)) * 100:.1f}%)")

        # Per disease top 15
        lines.append("")
        lines.append("### Top Canonical Diseases by Representation")
        disease_counts = valid_df["canonical_label"].value_counts().head(15)
        lines.append("")
        lines.append("| Canonical Label | Count |")
        lines.append("|-----------------|-------|")
        for label, cnt in disease_counts.items():
            lines.append(f"| `{label}` | {cnt} |")
    else:
        lines.append("*(No valid image data mounted yet. Dataset sources must be mounted or configured via environment variables.)*")

    lines.extend([
        "",
        "## 4. Integrity and Compliance Checklist",
        "",
        "- [x] No fabricated images or labels permitted.",
        "- [x] MD5 and perceptual dHash duplicate elimination active.",
        "- [x] PIL/OpenCV binary decode verification for corruptions.",
        "- [x] Strict unmapped label gating to `review/unmapped`.",
        "- [x] Source metadata retained for every record.",
        ""
    ])

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[+] Quality audit report written to: {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge and audit external crop disease datasets.")
    parser.add_argument("--sources-config", type=Path, default=Path(__file__).parent / "dataset_sources.json")
    parser.add_argument("--output-manifest", type=Path, default=Path(__file__).parent / "dataset_manifest.csv")
    parser.add_argument("--output-report", type=Path, default=Path(__file__).parent / "dataset_quality_check.md")
    parser.add_argument("--plantvillage-root", type=str, default=None)
    parser.add_argument("--plantdoc-root", type=str, default=None)
    parser.add_argument("--plantwild-root", type=str, default=None)
    parser.add_argument("--field-root", type=str, default=None)

    args = parser.parse_args()
    overrides = {
        "PlantVillage": args.plantvillage_root,
        "PlantDoc": args.plantdoc_root,
        "PlantWild": args.plantwild_root,
        "FieldData": args.field_root
    }
    run_merge(args.sources_config, args.output_manifest, args.output_report, overrides)

