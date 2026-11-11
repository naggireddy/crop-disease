"""
Leakage-Aware Dataset Splitting Module
Splits dataset into 70% Train, 15% Validation, 15% Test while guaranteeing
that duplicate or near-duplicate images (sharing MD5 or perceptual dHash)
NEVER leak across split boundaries.
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple
import pandas as pd
import numpy as np

def leakage_aware_split(
    manifest_path: Path,
    output_path: Path = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> pd.DataFrame:
    """
    Performs stratified, leakage-aware dataset splitting.
    Groups records by visual hash (MD5 / dHash) before assigning splits.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"
    output_path = output_path or manifest_path

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    if len(df) == 0:
        print("[!] Manifest is empty. Preserving schema.")
        if "split" not in df.columns:
            df["split"] = []
        df.to_csv(output_path, index=False)
        return df

    # Filter out corrupted or unmapped images from the core training split
    valid_mask = df["is_valid"] if "is_valid" in df.columns else pd.Series(True, index=df.index)
    if "status" in df.columns:
        valid_mask = valid_mask & (df["status"] != "unmapped")

    # Create duplicate grouping key based on dhash or md5_hash
    def get_group_key(row):
        dh = str(row.get("dhash", "")).strip()
        md = str(row.get("md5_hash", "")).strip()
        if dh and dh != "nan":
            return f"dh_{dh}"
        if md and md != "nan":
            return f"md_{md}"
        return f"id_{row.get('image_id', '')}"

    df["leakage_group"] = df.apply(get_group_key, axis=1)

    # Initialize split column
    df["split"] = "unassigned"

    # Split group keys per canonical_label to preserve class stratification
    rng = np.random.RandomState(seed)
    train_groups: Set[str] = set()
    val_groups: Set[str] = set()
    test_groups: Set[str] = set()

    label_col = "canonical_label" if "canonical_label" in df.columns else "crop"
    grouped = df[valid_mask].groupby(label_col)

    for label, group_df in grouped:
        unique_groups = group_df["leakage_group"].unique()
        rng.shuffle(unique_groups)

        n = len(unique_groups)
        if n == 1:
            # Single group belongs to train
            train_groups.add(unique_groups[0])
        elif n == 2:
            train_groups.add(unique_groups[0])
            val_groups.add(unique_groups[1])
        else:
            n_train = max(1, int(round(n * train_ratio)))
            n_val = max(1, int(round(n * val_ratio)))
            # Ensure total doesn't exceed n
            if n_train + n_val >= n:
                n_train = n - 2
                n_val = 1
            n_test = n - n_train - n_val

            for g in unique_groups[:n_train]:
                train_groups.add(g)
            for g in unique_groups[n_train:n_train + n_val]:
                val_groups.add(g)
            for g in unique_groups[n_train + n_val:]:
                test_groups.add(g)

    # Assign split to dataframe
    df.loc[df["leakage_group"].isin(train_groups), "split"] = "train"
    df.loc[df["leakage_group"].isin(val_groups), "split"] = "val"
    df.loc[df["leakage_group"].isin(test_groups), "split"] = "test"

    # Mark remaining invalid/unmapped images as excluded
    df.loc[~valid_mask, "split"] = "excluded"

    # Verification: Assert ZERO hash intersection across train, val, test
    train_h = set(df[df["split"] == "train"]["leakage_group"])
    val_h = set(df[df["split"] == "val"]["leakage_group"])
    test_h = set(df[df["split"] == "test"]["leakage_group"])

    assert len(train_h.intersection(val_h)) == 0, "Leakage detected between Train and Val!"
    assert len(train_h.intersection(test_h)) == 0, "Leakage detected between Train and Test!"
    assert len(val_h.intersection(test_h)) == 0, "Leakage detected between Val and Test!"

    # Clean temporary helper column
    df.drop(columns=["leakage_group"], inplace=True)

    # Save updated manifest
    df.to_csv(output_path, index=False)
    print(f"[+] Leakage-aware split completed and saved to {output_path}")
    print(f"    - Train: {len(df[df['split'] == 'train'])} images")
    print(f"    - Val:   {len(df[df['split'] == 'val'])} images")
    print(f"    - Test:  {len(df[df['split'] == 'test'])} images")
    print(f"    - Excluded (corrupt/unmapped): {len(df[df['split'] == 'excluded'])} images")

    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset manifest into Train, Val, Test.")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).parent / "dataset_manifest.csv")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    out = args.output or args.manifest
    leakage_aware_split(args.manifest, out, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)

