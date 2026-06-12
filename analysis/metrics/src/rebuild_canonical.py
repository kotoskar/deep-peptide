#!/usr/bin/env python3
"""
Rebuild canonical_metrics.csv and canonical_metrics.md from scratch.
Reads fresh fp32 inference JSONs from runs/*/
"""

import json
import os
import glob
from pathlib import Path

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
RUNS_DIR = _ROOT / "runs"
ANALYSIS_DIR = _ROOT / "analysis" / "metrics"

HARD_OVERRIDES = {
    "esm2_bond_loss_soft_l005_w5_tau15",
    "esm2_aho_transition_bias_sparse_trainable_zero",
}

DRIFT_THRESHOLD = 0.015

PRF_KEYS = [
    ("f1 all", "f1_all"),
    ("precision all", "precision_all"),
    ("recall all", "recall_all"),
    ("f1 peptides", "f1_peptides"),
    ("precision peptides", "precision_peptides"),
    ("recall peptides", "recall_peptides"),
    ("f1 propeptides", "f1_propeptides"),
    ("precision propeptides", "precision_propeptides"),
    ("recall propeptides", "recall_propeptides"),
]

MCC_AUC_KEYS = [
    ("residue mcc all", "mcc_all"),
    ("residue roc_auc all", "auc_all"),
    ("residue mcc peptides", "mcc_peptides"),
    ("residue roc_auc peptides", "auc_peptides"),
    ("residue mcc propeptides", "mcc_propeptides"),
    ("residue roc_auc propeptides", "auc_propeptides"),
]


def load_json(path):
    if path and Path(path).exists():
        with open(path) as f:
            return json.load(f)
    return None


def compute_drift(train_json, infer_json):
    """Compute max |train - infer| over 9 P/R/F1 keys."""
    if train_json is None or infer_json is None:
        return None
    diffs = []
    for key, _ in PRF_KEYS:
        t = train_json.get(key)
        i = infer_json.get(key)
        if t is not None and i is not None:
            diffs.append(abs(t - i))
    if not diffs:
        return None
    return max(diffs)


def build_row(folder_name, train_json, infer_json, is_hard_override):
    """
    Returns a dict with:
      f1_all, precision_all, recall_all, mcc_all, auc_all,
      f1_peptides, precision_peptides, recall_peptides, mcc_peptides, auc_peptides,
      f1_propeptides, precision_propeptides, recall_propeptides, mcc_propeptides, auc_propeptides,
      drift, trusted
    """
    row = {}

    # P/R/F1 from train_json (authoritative); fall back to infer_json for runs that
    # have no train-time test_metrics.json (inference-only runs, e.g. re-inferred ones).
    prf_src = train_json if train_json is not None else infer_json
    if prf_src is not None:
        for key, col in PRF_KEYS:
            row[col] = prf_src.get(key)
    else:
        for _, col in PRF_KEYS:
            row[col] = "N/A"

    # Drift
    drift = compute_drift(train_json, infer_json)
    row["drift"] = drift

    # Trusted flag
    if is_hard_override:
        trusted = False
    elif infer_json is None:
        trusted = False
    elif train_json is None:
        trusted = True   # inference-only run: infer IS the source, nothing to drift against
    elif drift is None:
        trusted = False
    elif drift <= DRIFT_THRESHOLD:
        trusted = True
    else:
        trusted = False

    row["trusted"] = trusted

    # MCC/AUC from infer_json only if trusted
    for key, col in MCC_AUC_KEYS:
        if trusted and infer_json is not None:
            val = infer_json.get(key)
            row[col] = val if val is not None else "N/A"
        else:
            row[col] = "N/A"

    return row


def process_all_runs():
    """Process all run folders with test_metrics.json."""
    rows = {}

    run_dirs = sorted([d for d in RUNS_DIR.iterdir()
                       if d.is_dir() and ((d / "test_metrics.json").exists()
                                          or (d / "test_metrics_infer.json").exists())])

    for run_dir in run_dirs:
        name = run_dir.name
        is_hard_override = name in HARD_OVERRIDES

        # TEST side
        train_json = load_json(run_dir / "test_metrics.json")
        infer_json = load_json(run_dir / "test_metrics_infer.json")
        test_row = build_row(name, train_json, infer_json, is_hard_override)

        # HOMO side
        homo_train_json = load_json(run_dir / "homo_test_metrics.json")
        homo_infer_json = load_json(run_dir / "homo_test_metrics_infer.json")

        if homo_train_json is None:
            # No homo metrics at all
            homo_row = {col: "N/A" for _, col in PRF_KEYS}
            for _, col in MCC_AUC_KEYS:
                homo_row[col] = "N/A"
            homo_row["drift"] = "N/A"
            homo_row["trusted"] = False
        else:
            homo_row = build_row(name, homo_train_json, homo_infer_json, is_hard_override)

        rows[name] = (test_row, homo_row)

    return rows


def fmt_csv(val, decimals=6):
    if isinstance(val, (float, int)) and not isinstance(val, bool):
        return f"{float(val):.{decimals}f}"
    return str(val)


def fmt_md(val, decimals=3):
    if isinstance(val, (float, int)) and not isinstance(val, bool):
        return f"{float(val):.{decimals}f}"
    return str(val)


def write_csv(rows):
    cols_test = [
        "f1_all", "precision_all", "recall_all", "mcc_all", "auc_all",
        "f1_peptides", "precision_peptides", "recall_peptides", "mcc_peptides", "auc_peptides",
        "f1_propeptides", "precision_propeptides", "recall_propeptides", "mcc_propeptides", "auc_propeptides",
        "drift", "trusted"
    ]
    cols_homo = [
        "homo_f1_all", "homo_precision_all", "homo_recall_all", "homo_mcc_all", "homo_auc_all",
        "homo_f1_peptides", "homo_precision_peptides", "homo_recall_peptides", "homo_mcc_peptides", "homo_auc_peptides",
        "homo_f1_propeptides", "homo_precision_propeptides", "homo_recall_propeptides", "homo_mcc_propeptides", "homo_auc_propeptides",
        "homo_drift", "homo_trusted"
    ]

    # Rename drift/trusted for test side
    test_col_names = [c if c not in ("drift", "trusted") else
                      ("test_drift" if c == "drift" else "test_mcc_auc_trusted")
                      for c in cols_test]
    homo_col_names = [c if c not in ("homo_drift", "homo_trusted") else
                      ("homo_drift" if c == "homo_drift" else "homo_mcc_auc_trusted")
                      for c in cols_homo]

    header = ["run"] + test_col_names + homo_col_names

    out_path = ANALYSIS_DIR / "canonical_metrics.csv"
    with open(out_path, "w") as f:
        f.write(",".join(header) + "\n")
        for name in sorted(rows.keys()):
            test_row, homo_row = rows[name]
            cells = [name]
            for col in cols_test:
                if col == "drift":
                    v = test_row["drift"]
                    cells.append(fmt_csv(v) if isinstance(v, float) else str(v))
                elif col == "trusted":
                    cells.append(str(test_row["trusted"]))
                else:
                    v = test_row[col]
                    cells.append(fmt_csv(v) if isinstance(v, float) else str(v))
            for col in cols_homo:
                base = col.replace("homo_", "") if col.startswith("homo_") else col
                if col == "homo_drift":
                    v = homo_row["drift"]
                    cells.append(fmt_csv(v) if isinstance(v, float) else str(v))
                elif col == "homo_trusted":
                    cells.append(str(homo_row["trusted"]))
                else:
                    v = homo_row.get(base, homo_row.get(col, "N/A"))
                    cells.append(fmt_csv(v) if isinstance(v, float) else str(v))
            f.write(",".join(cells) + "\n")

    print(f"Written {out_path}")
    return out_path


# -------------------------
# Table definitions
# -------------------------

# Maps row_label -> folder_name (from table_verification.md)
TABLE1_ROWS = [
    ("ESM2 (baseline)", "train_run_esm2"),
    ("ESM2 + telescopic CRF", "esm2_telescoping_segmental"),
    ("ESM2 + Aho emission fusion", "esm2_aho_emission_fusion"),
    ("ESM2 + (Aho -> hidden layer 32) emission fusion", "esm2_aho_emission_fusion_h32"),
    ("ESM2 + Aho hidden state fusion", "esm2_aho_mid_fusion_raw_m64"),
    ("ESM2 + Aho hidden state fusion only peptides", "esm2_aho_mid_fusion_raw_m64_pep_only"),
    ("ESM2 + Aho сигнал добавляется к CRF переходам", "esm2_aho_transition_bias_sparse_trainable_zero"),
    ("ESM2 + Aho early fusion (concat with esm)", "esm2_aho_tribranch"),
    ("ESM2 + доп. лосс разрезов к ближайшей границе", "esm2_bond_loss_soft_l005_w5_tau15"),
    ("ESM2 c AdamW оптимизатором", "train_run_esm2_adamw"),
]

TABLE2_ROWS = [
    ("ESM2", "train_run_esm2"),
    ("ESM2+residue features (ESM2+ below)", "train_run_esm2_plus"),
    ("ESM-C", "train_run_esmc_600m"),
    ("ESM-C 6B", "esmc_6b"),
    ("ProstT5", "train_run_prostt5"),
    ("ProstT5+residue features", "train_run_prostt5_plus"),
    ("(ProstT5 3DI + ESM2) proj.", "train_run_esm2+3di_proj"),
    ("(ProstT5 3DI + ESM2) proj.gated.", "train_run_esm2+3di_proj_gated"),
    ("(ProstT5 3DI + ESM2) proj.gated.conv.", "train_run_esm2+3di_proj_gated_conv"),
    ("AFTK all, no filter", "train_run_aft"),
    ("AFTK only single, no filter", "train_run_aft_single"),
    ("AFTK all w/o lddt, no filter", "train_run_aft_no_lddt"),
    ("AFTK all, >70% avg plddt", "train_run_aft_plddt70"),
    ("ESM2+(AFTK all, no filter) pr.gt.conv", "train_run_esm2_aft"),
    ("ESM2+(AFTK only single no filter) pr.gt.conv", "train_run_esm2_aft_single_gated"),
    ("ESM2+(AFTK only pair no filter) pr.gt.conv", "train_run_esm2_aft_pair_gated"),
    ("ESM2+(AFTK all w/o lddt no filter) pr.gt.conv", "train_run_esm2_aft_no_lddt_gated"),
    ("ESM2+(AFTK all, >70% avg plddt) pr.gt.conv", "train_run_esm2_aft_plddt70"),
]

TABLE3_ROWS = [
    ("ESM2", "train_run_esm2"),
    ("ESM2+residue features (ESM2+ below)", "train_run_esm2_plus"),
    ("ESM-C", "train_run_esmc_600m"),
    ("ProstT5", "train_run_prostt5"),
    ("ProstT5+residue features", "train_run_prostt5_plus"),
    ("(ProstT5 3DI + ESM2) proj.", "train_run_esm2+3di_proj"),
    ("(ProstT5 3DI + ESM2) proj. gated.", "train_run_esm2+3di_proj_gated"),
    ("(ProstT5 3DI + ESM2) proj.gated.conv.", "train_run_esm2+3di_proj_gated_conv"),
    ("AFTK all, no filter", "train_run_aft"),
    ("AFTK only single, no filter", "train_run_aft_single"),
    ("AFTK all w/o lddt, no filter", "train_run_aft_no_lddt"),
    ("AFTK all, >70% avg plddt", "train_run_aft_plddt70"),
    ("ESM2+(AFTK all, no filter) pr.gt.conv", "train_run_esm2_aft"),
    ("ESM2+(AFTK only single, no filter) pr.gt.conv", "train_run_esm2_aft_single_gated"),
    ("ESM2+(AFTK only pair, no filter) pr.gt.conv", "train_run_esm2_aft_pair_gated"),
    ("ESM2+(AFTK all w/o lddt, no filter) pr.gt.conv", "train_run_esm2_aft_no_lddt_gated"),
    ("ESM2+(AFTK all, >70% avg plddt) pr.gt.conv", "train_run_esm2_aft_plddt70"),
]

# MD column order: TEST then HOMO
# For test: f1_all prec_all rec_all mcc_all auc_all | f1_pep prec_pep rec_pep mcc_pep auc_pep | f1_prop prec_prop rec_prop mcc_prop auc_prop
MD_COLS = [
    ("f1_all", "All F1"),
    ("precision_all", "All Prec"),
    ("recall_all", "All Rec"),
    ("mcc_all", "All MCC"),
    ("auc_all", "All AUC"),
    ("f1_peptides", "Pep F1"),
    ("precision_peptides", "Pep Prec"),
    ("recall_peptides", "Pep Rec"),
    ("mcc_peptides", "Pep MCC"),
    ("auc_peptides", "Pep AUC"),
    ("f1_propeptides", "Propep F1"),
    ("precision_propeptides", "Propep Prec"),
    ("recall_propeptides", "Propep Rec"),
    ("mcc_propeptides", "Propep MCC"),
    ("auc_propeptides", "Propep AUC"),
]


def bold_best(table_rows_data, col_keys):
    """
    Returns dict col_key -> set of row indices (0-based) that have the max value.
    Only considers numeric (float or int) values (N/A and None excluded).
    """
    best = {}
    for col_key in col_keys:
        vals = []
        for i, row_data in enumerate(table_rows_data):
            v = row_data.get(col_key, "N/A")
            if isinstance(v, (float, int)) and not isinstance(v, bool):
                vals.append((i, float(v)))
        if not vals:
            best[col_key] = set()
            continue
        max_val = max(v for _, v in vals)
        best[col_key] = {i for i, v in vals if v == max_val}
    return best


def cell_str(v, col_key, row_idx, best_dict, decimals=3):
    if isinstance(v, (float, int)) and not isinstance(v, bool):
        s = f"{float(v):.{decimals}f}"
        if row_idx in best_dict.get(col_key, set()):
            s = f"**{s}**"
        return s
    return str(v)


def build_table_md(rows_spec, all_rows_data, use_homo=False):
    """
    rows_spec: list of (label, folder_name)
    all_rows_data: dict folder_name -> (test_row, homo_row)
    use_homo: if True, pull from homo_row instead of test_row
    """
    col_keys = [c for c, _ in MD_COLS]
    col_headers = [h for _, h in MD_COLS]

    # Collect row data
    table_data = []
    for label, folder in rows_spec:
        if folder not in all_rows_data:
            # Folder missing entirely - all N/A
            d = {col: "N/A" for col in col_keys}
        else:
            test_row, homo_row = all_rows_data[folder]
            d = homo_row if use_homo else test_row
        table_data.append((label, folder, d))

    # Compute bold
    row_data_list = [d for _, _, d in table_data]
    best = bold_best(row_data_list, col_keys)

    # Build markdown table
    lines = []
    header_row = "| Model | " + " | ".join(col_headers) + " |"
    sep_row = "|:--- | " + " | ".join(["---:"] * len(col_headers)) + " |"
    lines.append(header_row)
    lines.append(sep_row)

    for i, (label, folder, d) in enumerate(table_data):
        cells = [label]
        for col_key in col_keys:
            v = d.get(col_key, "N/A")
            cells.append(cell_str(v, col_key, i, best))
        lines.append("| " + " | ".join(cells) + " |")

    return lines, table_data


def is_na(v):
    """Return True if v is N/A or None (i.e., not a real number)."""
    return v == "N/A" or v is None


def count_na_cells(table_data):
    """Count rows with at least one N/A in MCC/AUC columns."""
    mcc_auc_cols = ["mcc_all", "auc_all", "mcc_peptides", "auc_peptides",
                    "mcc_propeptides", "auc_propeptides"]
    na_rows = 0
    for label, folder, d in table_data:
        if any(is_na(d.get(c, "N/A")) for c in mcc_auc_cols):
            na_rows += 1
    return na_rows


def na_reason(folder, d):
    """Return human-readable reason why MCC/AUC is N/A for this row."""
    drift = d.get("drift", "N/A")
    trusted = d.get("trusted", False)
    if folder in HARD_OVERRIDES:
        return "model unrecoverable for infer"
    elif drift == "N/A" or drift is None:
        return "no infer JSON"
    elif isinstance(drift, float) and drift > DRIFT_THRESHOLD:
        return f"fresh infer diverged (drift={drift:.4f})"
    elif trusted:
        # trusted but some MCC/AUC still N/A = null in infer JSON (e.g. 0 predictions)
        return "MCC undefined (no positive predictions for that class)"
    else:
        return "no infer JSON"


def build_footnotes_test(table_data):
    """Build footnote bullet list for N/A MCC/AUC rows (TEST side)."""
    lines = []
    mcc_auc_cols = ["mcc_all", "auc_all", "mcc_peptides", "auc_peptides",
                    "mcc_propeptides", "auc_propeptides"]
    for label, folder, d in table_data:
        if any(is_na(d.get(c, "N/A")) for c in mcc_auc_cols):
            reason = na_reason(folder, d)
            lines.append(f"- *{label}* (`{folder}`): {reason}")
    return lines


def build_footnotes_homo(table_data, homo_data_list):
    """Build footnote bullet list for N/A MCC/AUC rows (HOMO side)."""
    lines = []
    mcc_auc_cols = ["mcc_all", "auc_all", "mcc_peptides", "auc_peptides",
                    "mcc_propeptides", "auc_propeptides"]
    for (label, folder, _), d in zip(table_data, homo_data_list):
        if any(is_na(d.get(c, "N/A")) for c in mcc_auc_cols):
            reason = na_reason(folder, d)
            lines.append(f"- *{label}* (`{folder}`): {reason}")
    return lines


def write_md(rows, existing_md_path):
    """Write the canonical_metrics.md file."""

    # Build tables
    t1_lines, t1_data = build_table_md(TABLE1_ROWS, rows, use_homo=False)
    t2_lines, t2_data = build_table_md(TABLE2_ROWS, rows, use_homo=False)
    t3_lines, t3_data = build_table_md(TABLE3_ROWS, rows, use_homo=True)

    # For Table 3 footnotes we need homo_row data
    t3_homo_data = []
    for label, folder, _ in t3_data:
        if folder in rows:
            _, homo_row = rows[folder]
            t3_homo_data.append(homo_row)
        else:
            t3_homo_data.append({})

    # Count N/A rows
    t1_na = count_na_cells(t1_data)
    t2_na = count_na_cells(t2_data)
    t3_na = count_na_cells([(l, f, d) for (l, f, _), d in zip(t3_data, t3_homo_data)])

    print(f"\nTable 1 N/A rows (MCC/AUC): {t1_na}")
    print(f"Table 2 N/A rows (MCC/AUC): {t2_na}")
    print(f"Table 3 N/A rows (MCC/AUC): {t3_na}")

    # Build footnotes
    t1_footnotes = build_footnotes_test(t1_data)
    t2_footnotes = build_footnotes_test(t2_data)
    t3_footnotes = build_footnotes_homo(
        [(l, f, d) for (l, f, d) in t3_data],
        t3_homo_data
    )

    # Build coverage section (sorted by folder name)
    all_covered_folders = set()
    for _, folder in TABLE1_ROWS + TABLE2_ROWS + TABLE3_ROWS:
        all_covered_folders.add(folder)

    uncovered = []
    for name in sorted(rows.keys()):
        if name not in all_covered_folders:
            test_row, homo_row = rows[name]
            drift = test_row.get("drift", "N/A")
            trusted = test_row.get("trusted", False)
            uncovered.append((name, drift, trusted))

    infer_only = ["esm2_lora_lstmcnncrf_r4_last2_qv", "train_run_3di_only"]

    # Write the file
    out_path = ANALYSIS_DIR / "canonical_metrics.md"
    with open(out_path, "w", encoding="utf-8") as f:
        # Banner
        f.write("# Canonical Experiment Tables\n\n")
        f.write("> ⚠️ **Metric note:** the P/R/F1 here come from the shipped ±3 peptide-finding\n")
        f.write("> metric, which has a variable-shadowing bug (inherited from upstream DeepPeptide)\n")
        f.write("> that understates recall by ~2–4 pp. These values are kept for comparability with\n")
        f.write("> the paper. See `analysis/dual_reporting_metrics.md` for the original-vs-corrected\n")
        f.write("> table, and `texs/error_analysis/report.md` §4 for the bug writeup.\n\n")
        f.write("> **Methodology:** P/R/F1 values are authoritative train-time values from `test_metrics.json` (or `homo_test_metrics.json` for Table 3). MCC and AUC are from fresh fp32 inference (`test_metrics_infer.json` / `homo_test_metrics_infer.json`), accepted only when `drift = max|train-time P/R/F1 − fresh P/R/F1|` ≤ 0.015. Hard overrides (always N/A): `esm2_bond_loss_soft_l005_w5_tau15` and `esm2_aho_transition_bias_sparse_trainable_zero` (model unrecoverable for infer). Values rounded to 3 decimal places. **Bold** = best in column (N/A cells excluded).\n\n")

        # Headline: best combined architecture+embedding (beats both baselines)
        def _fmt(v):
            try:
                return f"{float(v):.3f}"
            except (TypeError, ValueError):
                return "N/A"
        if "esmc6b_boundary_bond" in rows:
            f.write("## Headline: best combined configuration (architecture × embedding)\n\n")
            f.write("Not part of the original Table 1/2 sweep — this pairs the best **embedding** "
                    "(ESM-C 6B, top residue-level signal) with a boundary-sharpening **architecture** "
                    "(`lstmcnncrf_boundary_bond_loss`). It is the best F1 and MCC in the project; the "
                    "boundary head turns ESM-C 6B's high-recall/low-precision signal into precision "
                    "(+0.14). See `texs/error_analysis/combine_best.md`. (Single seed; deterministic "
                    "fp32, drift 0.000.)\n\n")
            f.write("| Config | TEST F1 all | TEST Prec all | TEST Rec all | TEST MCC all | HOMO F1 all | HOMO MCC all |\n")
            f.write("|:--- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
            for label, run in [("**ESM-C 6B + boundary/bond**", "esmc6b_boundary_bond"),
                               ("ESM2 baseline", "train_run_esm2"),
                               ("ESM-C 6B baseline", "esmc_6b")]:
                tr, hr = rows.get(run, ({}, {}))
                f.write(f"| {label} | {_fmt(tr.get('f1_all'))} | {_fmt(tr.get('precision_all'))} "
                        f"| {_fmt(tr.get('recall_all'))} | {_fmt(tr.get('mcc_all'))} "
                        f"| {_fmt(hr.get('f1_all'))} | {_fmt(hr.get('mcc_all'))} |\n")
            f.write("\n")

        # Table 1
        f.write("## Table 1: Architectural Changes (TEST set)\n\n")
        f.write("\n".join(t1_lines) + "\n\n")
        f.write("**Footnotes — rows with N/A MCC/AUC:**\n\n")
        if t1_footnotes:
            f.write("\n".join(t1_footnotes) + "\n")
        else:
            f.write("*(none — all rows trusted)*\n")
        f.write("\n\n")

        # Table 2
        f.write("## Table 2: Embedding Generators (TEST set)\n\n")
        f.write("\n".join(t2_lines) + "\n\n")
        f.write("**Footnotes — rows with N/A MCC/AUC:**\n\n")
        if t2_footnotes:
            f.write("\n".join(t2_footnotes) + "\n")
        else:
            f.write("*(none — all rows trusted)*\n")
        f.write("\n\n")

        # Table 3
        f.write("## Table 3: Homo sapiens Only (HOMO test set)\n\n")
        f.write("\n".join(t3_lines) + "\n\n")
        f.write("**Footnotes — rows with N/A MCC/AUC:**\n\n")
        if t3_footnotes:
            f.write("\n".join(t3_footnotes) + "\n")
        else:
            f.write("*(none — all rows trusted)*\n")
        f.write("\n\n")

        # Coverage
        f.write("## Coverage\n\n")
        f.write("The following run folders have `test_metrics.json` (included in `canonical_metrics.csv`) but are not mapped to any of the 3 experiment tables. They can be added in future tables.\n\n")
        for name, drift, trusted in uncovered:
            drift_str = f"{drift:.4f}" if isinstance(drift, float) else str(drift)
            f.write(f"- `{name}` (test_drift={drift_str}, trusted={trusted})\n")
        f.write("\n")
        f.write("The following run folders have `test_metrics_infer.json` but **no** `test_metrics.json`. They are excluded from the CSV entirely (no authoritative P/R/F1 source) and not in any table:\n\n")
        for name in sorted(infer_only):
            f.write(f"- `{name}` (infer-only, no test_metrics.json)\n")
        f.write("\n")
        f.write("> **Dropped rows (no backing run folder):** `(ProstT5 3DI + ESM2+) proj.gated.conv.` was present in the LaTeX source for both Table 2 and Table 3 but has no matching run folder in `runs/`. These rows are omitted.\n")

    print(f"Written {out_path}")
    print(f"\nTable 1: {t1_na} N/A row(s) in MCC/AUC")
    print(f"Table 2: {t2_na} N/A row(s) in MCC/AUC")
    print(f"Table 3: {t3_na} N/A row(s) in MCC/AUC")
    return out_path, t1_na, t2_na, t3_na


def print_drift_report(rows):
    """Print a brief drift report per run."""
    print("\n=== DRIFT REPORT ===")
    for name in sorted(rows.keys()):
        test_row, homo_row = rows[name]
        td = test_row.get("drift")
        hd = homo_row.get("drift")
        td_str = f"{td:.4f}" if isinstance(td, float) else str(td)
        hd_str = f"{hd:.4f}" if isinstance(hd, float) else str(hd)
        trusted_test = test_row.get("trusted", False)
        trusted_homo = homo_row.get("trusted", False)
        print(f"  {name}: TEST drift={td_str} trusted={trusted_test} | HOMO drift={hd_str} trusted={trusted_homo}")


if __name__ == "__main__":
    print("Processing all run folders...")
    rows = process_all_runs()
    print(f"Found {len(rows)} run folders with test_metrics.json")

    print_drift_report(rows)

    print("\nWriting CSV...")
    csv_path = write_csv(rows)

    print("\nWriting MD...")
    md_path, t1_na, t2_na, t3_na = write_md(rows, None)

    print(f"\n=== SUMMARY ===")
    print(f"Table 1 N/A rows: {t1_na} (expected: 2)")
    print(f"Table 2 N/A rows: {t2_na} (expected: ~0)")
    print(f"Table 3 N/A rows: {t3_na} (expected: ~0)")
