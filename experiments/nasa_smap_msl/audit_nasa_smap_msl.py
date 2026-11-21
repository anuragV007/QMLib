"""
Full local audit for the NASA SMAP/MSL Telemanom anomaly dataset.

Run from the QMLib project root:
    python experiments/nasa_smap_msl/audit_nasa_smap_msl.py

The script reads:
    NASA Data/labeled_anomalies.csv
    NASA Data/data/data/train/*.npy
    NASA Data/data/data/test/*.npy

It writes:
    experiments/nasa_smap_msl/results/dataset_summary.json
    experiments/nasa_smap_msl/results/channel_profile.csv
    experiments/nasa_smap_msl/results/feature_profile.csv
    experiments/nasa_smap_msl/results/anomaly_intervals.csv
    experiments/nasa_smap_msl/results/telemanom_artifact_inventory.csv

It does not modify the source dataset or train any model.
"""

from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "NASA Data"
OUTPUT_DIR = (
    PROJECT_ROOT / "experiments" / "nasa_smap_msl" / "results"
)


def locate_data_dirs() -> tuple[Path, Path]:
    """Support the nested layout in this Kaggle archive and common alternatives."""
    candidates = [
        (DATA_ROOT / "data" / "data" / "train",
         DATA_ROOT / "data" / "data" / "test"),
        (DATA_ROOT / "data" / "train",
         DATA_ROOT / "data" / "test"),
    ]
    for train_dir, test_dir in candidates:
        if train_dir.is_dir() and test_dir.is_dir():
            return train_dir, test_dir

    raise FileNotFoundError(
        "Could not locate train/test folders. Checked:\n"
        + "\n".join(
            f"  {train} and {test}" for train, test in candidates
        )
    )


def resolve_column(df: pd.DataFrame, alternatives: list[str]) -> str:
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for name in alternatives:
        if name.lower() in lookup:
            return lookup[name.lower()]
    raise ValueError(
        f"Could not find any of {alternatives}. "
        f"Available columns: {list(df.columns)}"
    )


def parse_intervals(value: Any) -> list[tuple[int, int]]:
    """Parse anomaly ranges. Dataset convention is zero-based, inclusive."""
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []

    parsed = ast.literal_eval(value) if isinstance(value, str) else value
    if parsed is None:
        return []
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"Unexpected anomaly_sequences value: {value!r}")

    if len(parsed) == 2 and all(np.isscalar(x) for x in parsed):
        parsed = [parsed]

    output: list[tuple[int, int]] = []
    for pair in parsed:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"Invalid anomaly interval: {pair!r}")
        start, end = int(pair[0]), int(pair[1])
        if start < 0 or end < start:
            raise ValueError(f"Invalid anomaly interval: {pair!r}")
        output.append((start, end))
    return output


def merge_intervals(
    intervals: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Merge overlapping or adjacent zero-based inclusive intervals."""
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(int(a), int(b)) for a, b in merged]


def safe_corr(x: np.ndarray, lag: int) -> float:
    if len(x) <= lag or np.std(x[:-lag]) == 0 or np.std(x[lag:]) == 0:
        return float("nan")
    return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])


def finite_rows(x: np.ndarray) -> np.ndarray:
    return x[np.isfinite(x).all(axis=1)]


def row_uniqueness(x: np.ndarray) -> tuple[int, float]:
    if len(x) == 0:
        return 0, float("nan")
    count = int(np.unique(x, axis=0).shape[0])
    duplicate_pct = 100.0 * (1.0 - count / len(x))
    return count, duplicate_pct


def feature_stats(
    channel: str,
    split: str,
    x: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if x.ndim != 2:
        return rows

    for j in range(x.shape[1]):
        values = x[:, j]
        values = values[np.isfinite(values)]
        if len(values) == 0:
            continue

        unique_count = int(np.unique(values).size)
        binary = bool(
            np.all(np.isclose(values, np.round(values)))
            and np.all(np.isin(np.unique(values), [0.0, 1.0]))
        )

        rows.append({
            "channel": channel,
            "split": split,
            "feature_index": j,
            "role_by_format": (
                "target_telemetry" if j == 0
                else "command_or_context"
            ),
            "n_values": int(len(values)),
            "n_unique": unique_count,
            "is_constant": bool(unique_count == 1),
            "is_binary_0_1": binary,
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "q01": float(np.quantile(values, 0.01)),
            "median": float(np.median(values)),
            "q99": float(np.quantile(values, 0.99)),
            "max": float(np.max(values)),
            "nonzero_fraction": float(np.mean(values != 0)),
        })
    return rows


def inspect_precomputed_artifacts(
    data_root: Path,
    test_dir: Path,
) -> list[dict[str, Any]]:
    """Inventory the supplied Telemanom run without loading H5 models."""
    base = data_root / "data" / "data"
    if not base.exists():
        base = data_root / "data"

    results: list[dict[str, Any]] = []
    if not base.exists():
        return results

    for run_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        subdirs = {
            name: run_dir / name
            for name in ("models", "y_hat", "smoothed_errors")
        }
        if not any(path.is_dir() for path in subdirs.values()):
            continue

        model_files = (
            sorted(subdirs["models"].glob("*.h5"))
            if subdirs["models"].is_dir() else []
        )
        yhat_files = (
            sorted(subdirs["y_hat"].glob("*.npy"))
            if subdirs["y_hat"].is_dir() else []
        )
        error_files = (
            sorted(subdirs["smoothed_errors"].glob("*.npy"))
            if subdirs["smoothed_errors"].is_dir() else []
        )

        all_channels = sorted(
            {p.stem for p in model_files}
            | {p.stem for p in yhat_files}
            | {p.stem for p in error_files}
        )

        for channel in all_channels:
            test_file = test_dir / f"{channel}.npy"
            test_len = None
            if test_file.exists():
                try:
                    test_len = int(np.load(
                        test_file, mmap_mode="r", allow_pickle=False
                    ).shape[0])
                except Exception:
                    pass

            yhat_file = subdirs["y_hat"] / f"{channel}.npy"
            error_file = subdirs["smoothed_errors"] / f"{channel}.npy"

            def array_length(path: Path) -> int | None:
                if not path.exists():
                    return None
                try:
                    arr = np.load(path, mmap_mode="r", allow_pickle=False)
                    return int(arr.shape[0]) if arr.ndim else 1
                except Exception:
                    return None

            model_file = subdirs["models"] / f"{channel}.h5"
            results.append({
                "run_id": run_dir.name,
                "channel": channel,
                "has_h5_model": model_file.exists(),
                "h5_size_mb": (
                    round(model_file.stat().st_size / (1024 ** 2), 4)
                    if model_file.exists() else None
                ),
                "has_y_hat": yhat_file.exists(),
                "y_hat_length": array_length(yhat_file),
                "has_smoothed_errors": error_file.exists(),
                "smoothed_errors_length": array_length(error_file),
                "test_length": test_len,
            })

    return results


def main() -> None:
    train_dir, test_dir = locate_data_dirs()
    metadata_path = DATA_ROOT / "labeled_anomalies.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    metadata = pd.read_csv(metadata_path)

    channel_col = resolve_column(
        metadata, ["chan_id", "channel_id", "channel id"]
    )
    spacecraft_col = resolve_column(metadata, ["spacecraft"])
    intervals_col = resolve_column(
        metadata, ["anomaly_sequences", "anomaly sequences"]
    )

    class_col = next(
        (
            c for c in metadata.columns
            if str(c).strip().lower() == "class"
        ),
        None,
    )
    num_values_col = next(
        (
            c for c in metadata.columns
            if str(c).strip().lower() in {"num_values", "num values"}
        ),
        None,
    )

    train_files = {p.stem: p for p in train_dir.glob("*.npy")}
    test_files = {p.stem: p for p in test_dir.glob("*.npy")}
    all_channels = sorted(set(train_files) | set(test_files))

    metadata_by_channel: dict[str, pd.DataFrame] = {
        str(channel): group.copy()
        for channel, group in metadata.groupby(channel_col, dropna=False)
    }

    channel_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []

    totals = {
        "train_timesteps": 0,
        "test_timesteps": 0,
        "train_channels_loaded": 0,
        "test_channels_loaded": 0,
        "channels_with_metadata": 0,
        "metadata_rows": int(len(metadata)),
    }

    spacecraft_channel_counts: dict[str, int] = {}
    class_interval_counts: dict[str, int] = {}
    class_point_counts: dict[str, int] = {}

    for channel in all_channels:
        train = (
            np.asarray(np.load(train_files[channel], allow_pickle=False))
            if channel in train_files else None
        )
        test = (
            np.asarray(np.load(test_files[channel], allow_pickle=False))
            if channel in test_files else None
        )

        if train is not None and train.ndim == 2:
            totals["train_channels_loaded"] += 1
            totals["train_timesteps"] += int(train.shape[0])
            train_finite = finite_rows(train)
            feature_rows.extend(
                feature_stats(channel, "train", train_finite)
            )
            train_unique, train_duplicate_pct = row_uniqueness(train_finite)
        else:
            train_finite = np.empty((0, 0))
            train_unique, train_duplicate_pct = None, None

        if test is not None and test.ndim == 2:
            totals["test_channels_loaded"] += 1
            totals["test_timesteps"] += int(test.shape[0])
            test_finite = finite_rows(test)
            feature_rows.extend(
                feature_stats(channel, "test", test_finite)
            )
            test_unique, test_duplicate_pct = row_uniqueness(test_finite)
        else:
            test_finite = np.empty((0, 0))
            test_unique, test_duplicate_pct = None, None

        meta_rows = metadata_by_channel.get(channel)
        intervals: list[tuple[int, int]] = []
        class_values: list[str] = []
        spacecraft_values: list[str] = []
        expected_lengths: list[int] = []

        if meta_rows is not None:
            totals["channels_with_metadata"] += 1
            for _, row in meta_rows.iterrows():
                intervals.extend(parse_intervals(row[intervals_col]))

                spacecraft_values.append(str(row[spacecraft_col]))
                if class_col is not None and pd.notna(row[class_col]):
                    class_values.append(str(row[class_col]).strip())

                if num_values_col is not None and pd.notna(row[num_values_col]):
                    expected_lengths.append(int(row[num_values_col]))

                row_intervals = parse_intervals(row[intervals_col])
                cls = (
                    str(row[class_col]).strip()
                    if class_col is not None and pd.notna(row[class_col])
                    else "unspecified"
                )
                for start, end in row_intervals:
                    interval_rows.append({
                        "channel": channel,
                        "spacecraft": str(row[spacecraft_col]),
                        "class": cls,
                        "start_index": start,
                        "end_index_inclusive": end,
                        "length_points": end - start + 1,
                    })
                    class_interval_counts[cls] = (
                        class_interval_counts.get(cls, 0) + 1
                    )
                    class_point_counts[cls] = (
                        class_point_counts.get(cls, 0) + end - start + 1
                    )

        merged = merge_intervals(intervals)
        labelled_points = sum(end - start + 1 for start, end in merged)
        test_length = int(test.shape[0]) if test is not None and test.ndim == 2 else None
        out_of_range = 0
        if test_length is not None:
            out_of_range = sum(
                1 for start, end in merged
                if end >= test_length
            )

        spacecraft = (
            sorted(set(spacecraft_values))[0]
            if spacecraft_values else "unlisted"
        )
        if len(set(spacecraft_values)) > 1:
            spacecraft = "|".join(sorted(set(spacecraft_values)))

        spacecraft_channel_counts[spacecraft] = (
            spacecraft_channel_counts.get(spacecraft, 0) + 1
        )

        target = train_finite[:, 0] if train_finite.size else np.array([])
        context = train_finite[:, 1:] if train_finite.ndim == 2 and train_finite.shape[1] > 1 else np.empty((0, 0))

        if len(target):
            target_unique = int(np.unique(target).size)
            target_std = float(np.std(target))
            target_diff_std = (
                float(np.std(np.diff(target))) if len(target) > 1 else float("nan")
            )
            target_change_fraction = (
                float(np.mean(np.diff(target) != 0))
                if len(target) > 1 else float("nan")
            )
            target_acf1 = safe_corr(target, 1)
        else:
            target_unique = 0
            target_std = float("nan")
            target_diff_std = float("nan")
            target_change_fraction = float("nan")
            target_acf1 = float("nan")

        variable_context = (
            int(sum(np.unique(context[:, j]).size > 1 for j in range(context.shape[1])))
            if context.size else 0
        )
        binary_context = (
            int(sum(
                np.all(np.isclose(context[:, j], np.round(context[:, j])))
                and np.all(np.isin(np.unique(context[:, j]), [0.0, 1.0]))
                for j in range(context.shape[1])
            ))
            if context.size else 0
        )
        any_context_active_rate = (
            float(np.mean(np.any(context != 0, axis=1)))
            if context.size else float("nan")
        )

        channel_rows.append({
            "channel": channel,
            "spacecraft": spacecraft,
            "metadata_rows": int(len(meta_rows)) if meta_rows is not None else 0,
            "classes": "|".join(sorted(set(class_values))),
            "n_anomaly_intervals_raw": int(len(intervals)),
            "n_anomaly_intervals_merged": int(len(merged)),
            "labelled_test_points_merged": int(labelled_points),
            "test_label_prevalence": (
                float(labelled_points / test_length)
                if test_length else float("nan")
            ),
            "label_intervals_out_of_range": int(out_of_range),
            "metadata_num_values_match_test": (
                bool(all(n == test_length for n in expected_lengths))
                if expected_lengths and test_length is not None else None
            ),
            "train_shape": str(train.shape) if train is not None else None,
            "test_shape": str(test.shape) if test is not None else None,
            "train_finite_rows": int(len(train_finite)),
            "test_finite_rows": int(len(test_finite)),
            "train_unique_full_rows": train_unique,
            "train_duplicate_full_row_pct": train_duplicate_pct,
            "test_unique_full_rows": test_unique,
            "test_duplicate_full_row_pct": test_duplicate_pct,
            "target_unique_train": target_unique,
            "target_std_train": target_std,
            "target_diff_std_train": target_diff_std,
            "target_change_fraction_train": target_change_fraction,
            "target_lag1_corr_train": target_acf1,
            "variable_context_features_train": variable_context,
            "binary_context_features_train": binary_context,
            "context_active_row_fraction_train": any_context_active_rate,
        })

    channel_df = pd.DataFrame(channel_rows)
    feature_df = pd.DataFrame(feature_rows)
    interval_df = pd.DataFrame(interval_rows)
    artifact_rows = inspect_precomputed_artifacts(DATA_ROOT, test_dir)
    artifact_df = pd.DataFrame(artifact_rows)

    if not channel_df.empty:
        # Sort for inspection only; no test labels or performance used.
        channel_df = channel_df.sort_values(
            [
                "spacecraft",
                "target_unique_train",
                "target_std_train",
                "variable_context_features_train",
            ],
            ascending=[True, False, False, False],
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    channel_path = OUTPUT_DIR / "channel_profile.csv"
    feature_path = OUTPUT_DIR / "feature_profile.csv"
    interval_path = OUTPUT_DIR / "anomaly_intervals.csv"
    artifact_path = OUTPUT_DIR / "telemanom_artifact_inventory.csv"
    summary_path = OUTPUT_DIR / "dataset_summary.json"

    channel_df.to_csv(channel_path, index=False)
    feature_df.to_csv(feature_path, index=False)
    interval_df.to_csv(interval_path, index=False)
    artifact_df.to_csv(artifact_path, index=False)

    # Compare file sets to catch missing train/test partners.
    train_only = sorted(set(train_files) - set(test_files))
    test_only = sorted(set(test_files) - set(train_files))
    metadata_only = sorted(set(metadata_by_channel) - set(all_channels))
    arrays_without_metadata = sorted(set(all_channels) - set(metadata_by_channel))

    summary = {
        "dataset_name": "NASA SMAP/MSL Telemanom anomaly benchmark",
        "data_root": str(DATA_ROOT),
        "train_dir": str(train_dir),
        "test_dir": str(test_dir),
        "metadata_file": str(metadata_path),
        "metadata_columns": [str(c) for c in metadata.columns],
        "train_array_files": len(train_files),
        "test_array_files": len(test_files),
        "unique_array_channels": len(all_channels),
        "train_only_channels": train_only,
        "test_only_channels": test_only,
        "metadata_only_channels": metadata_only,
        "array_channels_without_metadata": arrays_without_metadata,
        "totals": totals,
        "spacecraft_channel_counts_from_metadata": spacecraft_channel_counts,
        "anomaly_interval_counts_by_class": class_interval_counts,
        "anomaly_points_by_class_before_merging": class_point_counts,
        "precomputed_telemanom_run_ids": sorted(
            artifact_df["run_id"].unique().tolist()
        ) if not artifact_df.empty else [],
        "precomputed_artifact_rows": int(len(artifact_df)),
        "notes": [
            "Anomaly intervals use zero-based inclusive endpoints.",
            "Feature 0 is the telemetry target in the Telemanom input convention.",
            "Remaining columns are command/context inputs in the Telemanom format.",
            "No model was trained and no input data was modified by this audit.",
            "Labels are attached to test sequences; they must not be used for fitting or threshold selection.",
        ],
        "outputs": {
            "channel_profile": str(channel_path),
            "feature_profile": str(feature_path),
            "anomaly_intervals": str(interval_path),
            "telemanom_artifact_inventory": str(artifact_path),
        },
    }

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nNASA SMAP/MSL DATASET AUDIT")
    print("=" * 88)
    print("Train arrays:", len(train_files))
    print("Test arrays:", len(test_files))
    print("Metadata rows:", len(metadata))
    print("Array channels:", len(all_channels))
    print("Total train timesteps:", totals["train_timesteps"])
    print("Total test timesteps:", totals["test_timesteps"])
    print("Channels with metadata:", totals["channels_with_metadata"])
    print("Train-only channels:", train_only)
    print("Test-only channels:", test_only)
    print("Metadata-only channels:", metadata_only)
    print("Class interval counts:", class_interval_counts)

    if not channel_df.empty:
        show_cols = [
            "channel",
            "spacecraft",
            "train_shape",
            "test_shape",
            "target_unique_train",
            "target_std_train",
            "variable_context_features_train",
            "n_anomaly_intervals_merged",
            "test_label_prevalence",
        ]
        print("\nCHANNEL PROFILE (first 30 rows, sorted by target variation)")
        print(channel_df[show_cols].head(30).to_string(index=False))

    print("\nSaved audit reports:")
    for path in (summary_path, channel_path, feature_path, interval_path, artifact_path):
        print(path)


if __name__ == "__main__":
    main()
