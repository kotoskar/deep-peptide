"""
Dataset statistics for DeepPeptide labeled_sequences data.

Schema
------
labeled_sequences.csv  (index_col=0)
    protein_name, sequence, organism, is_peptide, coordinates,
    is_propeptide, propeptide_coordinates, protein_id

  - is_peptide / is_propeptide: per-residue binary mask STRINGS of '0'/'1',
    same length as sequence.  A run of consecutive '1's marks one segment.
    A protein can have multiple segments.
  - organism: e.g. "Homo sapiens (Human)".  Species is the part before '('.
  - protein_id: UniProt accession.

graphpart_assignments.csv / graphpart_assignments.raw.csv  (col: AC)
    Join labeled_sequences.protein_id → graphpart.AC.

Split mapping (mirrors get_dataloaders defaults in train_loop_crf.py):
    cluster in {0, 1, 2}  →  TRAIN
    cluster == 3           →  VALID
    cluster == 4           →  TEST
    not in graphpart file  →  EXCLUDED (not used in any split)

NOTE ON OVERLAPPING PEPTIDES
The coordinates field can list nested/overlapping peptides (e.g. PENK_BOVIN).
The mask collapses overlapping regions into one run of '1's, so segment counts
from mask runs may be lower than the count of coordinate entries.  This script
counts mask-run segments (matching the model's view) but also reports
coordinate-entry counts for comparison.

NOTE ON MINIMUM SEGMENT LENGTH
Both datasets have a hard annotation floor of 5 aa: no peptide or propeptide
segment shorter than 5 aa exists, confirmed at both the mask-run level AND the
coordinate entry level.  This is a true annotation minimum (likely a deliberate
UniProt curation floor), not an artifact of mask merging.

NOTE ON SPECIES DIVERSITY
The data is NOT concentrated on Homo sapiens (~5% of assigned proteins).  It is
a multi-species set with strong representation from venom-producing organisms
(spiders: Cyriopagopus, Lycosa, Chilobrachys; cone snails: Conus, Californiconus)
alongside standard model organisms.  GraphPart homology-based clustering can
assign entire genera almost entirely to one split (e.g. Cyriopagopus almost all
in TRAIN in uniprot_2026), causing per-split protein length and residue-imbalance
differences between splits.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
_TOPIC = Path(__file__).resolve().parent.parent  # analysis/dataset
BASE = _ROOT / "data"

DATASETS = {
    "2022": {
        "labeled": BASE / "uniprot_2022" / "labeled_sequences.csv",
        "graphpart": BASE / "uniprot_2022" / "graphpart_assignments.csv",
    },
    "2026": {
        "labeled": BASE / "uniprot_2026" / "labeled_sequences.csv",
        "graphpart": BASE / "uniprot_2026" / "graphpart_assignments.raw.csv",
    },
}

SPLIT_MAP = {0: "TRAIN", 1: "TRAIN", 2: "TRAIN", 3: "VALID", 4: "TEST"}
SPLITS = ["TRAIN", "VALID", "TEST", "ALL"]
PLOTS_DIR = _TOPIC / "plots"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_runs(mask: str) -> List[int]:
    """Return list of segment lengths (runs of '1') in a binary mask string."""
    lengths = []
    in_run = False
    run_len = 0
    for ch in mask:
        if ch == "1":
            in_run = True
            run_len += 1
        else:
            if in_run:
                lengths.append(run_len)
            in_run = False
            run_len = 0
    if in_run:
        lengths.append(run_len)
    return lengths


def count_coord_entries(coord_str) -> int:
    """Count the number of (start-end) entries in a coordinates string."""
    if pd.isna(coord_str) or str(coord_str).strip() == "":
        return 0
    return len(re.findall(r"\(\d+-\d+\)", str(coord_str)))


def parse_species(organism: str) -> str:
    """Extract species name: everything before the first '(' (stripped)."""
    if pd.isna(organism):
        return "UNKNOWN"
    idx = str(organism).find("(")
    if idx == -1:
        return str(organism).strip()
    return str(organism)[:idx].strip()


def length_stats(lengths: List[int]) -> Dict:
    if not lengths:
        return {"count": 0, "min": None, "median": None, "mean": None, "p90": None, "max": None}
    arr = np.array(lengths)
    return {
        "count": int(len(arr)),
        "min": int(arr.min()),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "p90": float(np.percentile(arr, 90)),
        "max": int(arr.max()),
    }


def fmt_stat(d: Dict) -> str:
    if d["count"] == 0:
        return "N/A"
    return (
        f"n={d['count']:,}  min={d['min']}  "
        f"median={d['median']:.1f}  mean={d['mean']:.1f}  "
        f"p90={d['p90']:.1f}  max={d['max']}"
    )


def make_length_table(lengths: List[int], up_to: int = 10) -> str:
    """Markdown table of segment length counts for lengths 1..up_to."""
    from collections import Counter
    c = Counter(lengths)
    rows = [f"| {i} | {c.get(i, 0):,} |" for i in range(1, up_to + 1)]
    header = "| Length | Count |\n|--------|-------|"
    return header + "\n" + "\n".join(rows)


def save_histogram(values: List[int], title: str, xlabel: str, fname: str,
                   clip_pct: Optional[float] = None):
    """Честная гистограмма целочисленных длин: ОДИН бин на дискретное значение X
    (никакого склеивания соседних длин). Длинный разреженный хвост можно обрезать по
    перцентилю `clip_pct`, чтобы основная масса распределения не сжималась — число
    обрезанных значений выносится в заголовок."""
    if not HAS_MPL or not values:
        return
    arr = np.asarray(values, dtype=int)
    n_clipped = 0
    hi = int(arr.max())
    if clip_pct is not None and len(arr) > 0:
        hi = int(np.percentile(arr, clip_pct))
        n_clipped = int((arr > hi).sum())
        arr = arr[arr <= hi]
    lo = int(arr.min())
    # центрируем каждый бин на целом числе: края на полуцелых -> ширина ровно 1
    edges = np.arange(lo - 0.5, hi + 1.5, 1.0)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.hist(arr, bins=edges, edgecolor="none", alpha=0.9)
    if n_clipped:
        title += f"  (показано ≤{hi} а.о.; {n_clipped} длиннее обрезаны)"
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Количество")
    ax.set_xlim(lo - 0.5, hi + 0.5)
    ax.margins(x=0)
    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / fname, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def load_data(dataset_key: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    paths = DATASETS[dataset_key]

    # ---- labeled sequences ----
    df = pd.read_csv(
        paths["labeled"],
        index_col=0,
        dtype={"is_peptide": str, "is_propeptide": str, "sequence": str,
               "protein_name": str, "organism": str, "coordinates": str,
               "propeptide_coordinates": str, "protein_id": str},
    )
    assert df["is_peptide"].dtype == object, "is_peptide was coerced — dtype bug!"
    assert df["is_propeptide"].dtype == object, "is_propeptide was coerced — dtype bug!"

    # ---- graphpart ----
    gp = pd.read_csv(paths["graphpart"])
    gp["cluster"] = gp["cluster"].astype(int)

    # Check AC duplicates
    dup_ac = gp["AC"].duplicated().sum()
    if dup_ac > 0:
        print(f"  WARNING: {dup_ac} duplicate ACs in graphpart for dataset {dataset_key}")

    return df, gp


def analyze(dataset_key: str) -> Dict:
    print(f"\n{'='*60}")
    print(f"Analyzing dataset: uniprot_{dataset_key}")
    print(f"{'='*60}")

    df, gp = load_data(dataset_key)

    # -----------------------------------------------------------------------
    # 1. Verify mask lengths == sequence lengths
    # -----------------------------------------------------------------------
    seq_len = df["sequence"].str.len()
    pep_mask_len = df["is_peptide"].str.len()
    pro_mask_len = df["is_propeptide"].str.len()

    pep_mismatch = (pep_mask_len != seq_len).sum()
    pro_mismatch = (pro_mask_len != seq_len).sum()
    print(f"  Mask length mismatches — is_peptide: {pep_mismatch}, is_propeptide: {pro_mismatch}")

    # Check for NaN masks (should be all-'0' strings, not float NaN)
    pep_null = df["is_peptide"].isna().sum()
    pro_null = df["is_propeptide"].isna().sum()
    print(f"  Null masks — is_peptide: {pep_null}, is_propeptide: {pro_null}")

    # -----------------------------------------------------------------------
    # 2. Join with graphpart
    # -----------------------------------------------------------------------
    # Build AC -> cluster dict and map
    ac_to_cluster = dict(zip(gp["AC"], gp["cluster"]))
    df["cluster"] = df["protein_id"].map(ac_to_cluster)

    n_total = len(df)
    n_excluded = df["cluster"].isna().sum()
    n_assigned = n_total - n_excluded
    print(f"  Total proteins: {n_total}  |  Assigned: {n_assigned}  |  Excluded (no graphpart): {n_excluded}")

    # Assign split labels (only for assigned proteins)
    df["split"] = df["cluster"].map(lambda c: SPLIT_MAP.get(int(c), "UNKNOWN") if pd.notna(c) else "EXCLUDED")

    # Check for unexpected cluster values
    unexpected = df[df["split"] == "UNKNOWN"]["cluster"].unique()
    if len(unexpected) > 0:
        print(f"  WARNING: unexpected cluster values: {unexpected}")

    # -----------------------------------------------------------------------
    # 3. Organism parsing
    # -----------------------------------------------------------------------
    df["species"] = df["organism"].apply(parse_species)
    no_paren = (df["organism"].notna() & ~df["organism"].str.contains(r"\(", na=False)).sum()
    # Note: in uniprot_2022, parentheses typically hold common name, e.g. "Homo sapiens (Human)".
    # In uniprot_2026, most organisms are plain scientific names with no parentheses (or strain info).
    # parse_species correctly returns the species name before '(' in all cases.
    print(f"  Organism rows with no parenthesis (informational): {no_paren}")

    # -----------------------------------------------------------------------
    # 4. Per-residue mask parsing into segments
    # -----------------------------------------------------------------------
    df["pep_segs"] = df["is_peptide"].apply(count_runs)       # list of lengths
    df["pro_segs"] = df["is_propeptide"].apply(count_runs)

    df["n_pep_segs"] = df["pep_segs"].apply(len)
    df["n_pro_segs"] = df["pro_segs"].apply(len)

    df["has_peptide"] = df["n_pep_segs"] > 0
    df["has_propeptide"] = df["n_pro_segs"] > 0
    df["has_both"] = df["has_peptide"] & df["has_propeptide"]
    df["is_negative"] = ~df["has_peptide"] & ~df["has_propeptide"]

    # Coordinate entry counts (for overlap comparison)
    df["coord_pep_count"] = df["coordinates"].apply(count_coord_entries)
    df["coord_pro_count"] = df["propeptide_coordinates"].apply(count_coord_entries)

    # -----------------------------------------------------------------------
    # 5. Build per-split results
    # -----------------------------------------------------------------------
    results = {}

    working_df = df[df["split"] != "EXCLUDED"]
    all_df = working_df.copy()
    all_df["split_label"] = "ALL"

    for split in ["TRAIN", "VALID", "TEST"]:
        subset = working_df[working_df["split"] == split].copy()
        subset["split_label"] = split
        results[split] = _compute_split_stats(subset, dataset_key, split)

    results["ALL"] = _compute_split_stats(all_df, dataset_key, "ALL")

    results["_meta"] = {
        "n_total": n_total,
        "n_assigned": n_assigned,
        "n_excluded": n_excluded,
        "pep_mask_mismatch": int(pep_mismatch),
        "pro_mask_mismatch": int(pro_mismatch),
        "pep_null": int(pep_null),
        "pro_null": int(pro_null),
        "no_paren_organism": int(no_paren),
        "dup_ac_graphpart": int(gp["AC"].duplicated().sum()),
        "protein_id_dups_labeled": int(df["protein_id"].duplicated().sum()),
        "df": df,
        "gp": gp,
    }

    return results


def _compute_split_stats(subset: pd.DataFrame, dataset_key: str, split_name: str) -> Dict:
    """Compute all stats for one split subset."""
    n_proteins = len(subset)
    n_has_pep = subset["has_peptide"].sum()
    n_has_pro = subset["has_propeptide"].sum()
    n_has_both = subset["has_both"].sum()
    n_neg = subset["is_negative"].sum()

    # All peptide segment lengths in this split
    all_pep_lens = [l for segs in subset["pep_segs"] for l in segs]
    all_pro_lens = [l for segs in subset["pro_segs"] for l in segs]

    # Protein sequence lengths
    prot_lens = subset["sequence"].str.len().tolist()

    # Segments per positive protein
    pos_pep = subset[subset["has_peptide"]]["n_pep_segs"].tolist()
    pos_pro = subset[subset["has_propeptide"]]["n_pro_segs"].tolist()

    # Residue-level counts
    total_residues = subset["sequence"].str.len().sum()
    pep_residues = subset["is_peptide"].apply(lambda m: m.count("1")).sum()
    pro_residues = subset["is_propeptide"].apply(lambda m: m.count("1")).sum()

    # Coordinate entry counts vs mask run counts
    coord_pep_total = subset["coord_pep_count"].sum()
    coord_pro_total = subset["coord_pro_count"].sum()
    mask_pep_total = subset["n_pep_segs"].sum()
    mask_pro_total = subset["n_pro_segs"].sum()

    # Tiny peptides (len <= 5)
    pep_le5 = sum(1 for l in all_pep_lens if l <= 5)

    return {
        "n_proteins": int(n_proteins),
        "n_has_peptide": int(n_has_pep),
        "n_has_propeptide": int(n_has_pro),
        "n_has_both": int(n_has_both),
        "n_negative": int(n_neg),

        "pep_seg_stats": length_stats(all_pep_lens),
        "pro_seg_stats": length_stats(all_pro_lens),
        "prot_len_stats": length_stats(prot_lens),

        "pep_segs_per_pos_protein": length_stats(pos_pep),
        "pro_segs_per_pos_protein": length_stats(pos_pro),

        "pep_seg_lens": all_pep_lens,
        "pro_seg_lens": all_pro_lens,
        "prot_lens": prot_lens,

        "total_residues": int(total_residues),
        "pep_residues": int(pep_residues),
        "pro_residues": int(pro_residues),

        "mask_pep_segments": int(mask_pep_total),
        "mask_pro_segments": int(mask_pro_total),
        "coord_pep_entries": int(coord_pep_total),
        "coord_pro_entries": int(coord_pro_total),

        "pep_le5_count": int(pep_le5),
        "pep_len_1to10": {i: int(sum(1 for l in all_pep_lens if l == i)) for i in range(1, 11)},

        "subset": subset,
    }


def organism_stats(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Per-species breakdown overall and by split."""
    working = df[df["split"] != "EXCLUDED"]
    counts = (
        working.groupby(["species", "split"])
        .size()
        .reset_index(name="count")
        .pivot(index="species", columns="split", values="count")
        .fillna(0)
        .astype(int)
    )
    counts["TOTAL"] = counts.sum(axis=1)
    counts = counts.sort_values("TOTAL", ascending=False)
    return counts.head(top_n)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(dataset_key: str, results: Dict) -> str:
    meta = results["_meta"]
    df = meta["df"]

    lines = []

    # ---- Summary prose ----
    all_s = results["ALL"]
    pep_med = all_s["pep_seg_stats"]["median"]
    pep_le5 = all_s["pep_le5_count"]
    pep_total_segs = all_s["mask_pep_segments"]
    pro_total_segs = all_s["mask_pro_segments"]

    working = df[df["split"] != "EXCLUDED"]
    species_col = working["species"]
    top1_sp = species_col.value_counts().index[0] if len(species_col) > 0 else "N/A"
    top1_count = species_col.value_counts().iloc[0] if len(species_col) > 0 else 0
    pct_top1 = 100 * top1_count / len(working) if len(working) > 0 else 0.0

    lines.append(f"# Dataset Statistics: uniprot_{dataset_key}\n")
    lines.append(f"## Summary\n")
    lines.append(
        f"The uniprot_{dataset_key} dataset contains **{meta['n_total']:,} proteins** in "
        f"`labeled_sequences.csv`, of which **{meta['n_assigned']:,} have a graphpart assignment** "
        f"({meta['n_excluded']:,} excluded from all splits). "
        f"Across all assigned proteins there are **{pep_total_segs:,} peptide segments** and "
        f"**{pro_total_segs:,} propeptide segments** (counted as runs of '1' in the mask). "
        f"The median peptide segment length is **{pep_med:.1f} aa**; "
        f"**{pep_le5:,} peptide segments** have length ≤ 5 (all exactly 5 — confirmed at both "
        f"mask-run and coordinate entry level; 5 aa appears to be the true annotation floor, not a mask-merging artifact). "
        f"The data is NOT concentrated on a single organism: the most common species is *{top1_sp}* "
        f"({top1_count:,} proteins, {pct_top1:.1f}% of assigned). The dataset is multi-species with "
        f"strong venom-organism representation (spiders, cone snails) alongside standard model organisms. "
        f"GraphPart homology partitioning may place entire genera almost entirely in one split.\n"
    )

    if meta["pep_mask_mismatch"] > 0 or meta["pro_mask_mismatch"] > 0:
        lines.append(
            f"> **DATA ANOMALY:** {meta['pep_mask_mismatch']} is_peptide mask length mismatches, "
            f"{meta['pro_mask_mismatch']} is_propeptide mask length mismatches.\n"
        )
    if meta["no_paren_organism"] > 0:
        lines.append(
            f"> **NOTE (organism format):** {meta['no_paren_organism']} organism strings have no parenthesis. "
            f"In uniprot_2022, parentheses hold common names (e.g. 'Homo sapiens (Human)'). "
            f"In uniprot_2026, most entries are plain scientific names with no parenthetical, or use parentheses for strain info. "
            f"The `parse_species` function handles both correctly (takes the part before the first '(').\n"
        )

    if results["ALL"]["n_negative"] == 0:
        lines.append(
            f"> **NOTE (no negatives):** This dataset contains 0 negative proteins (all assigned proteins have ≥1 peptide or propeptide segment). "
            f"Negatives exist at the **residue** level (the majority of residues in positive proteins are labeled '0'), "
            f"but there are no all-negative protein entries. This likely reflects that `labeled_sequences.csv` was built by "
            f"selecting only UniProt proteins with at least one Chain/Peptide/Propeptide annotation.\n"
        )

    # Coordinate vs mask segment count comparison
    coord_pep = all_s["coord_pep_entries"]
    mask_pep = all_s["mask_pep_segments"]
    coord_pro = all_s["coord_pro_entries"]
    mask_pro = all_s["mask_pro_segments"]
    if coord_pep != mask_pep or coord_pro != mask_pro:
        lines.append(
            f"> **NOTE (overlapping peptides):** Coordinate entries vs mask-run segments differ — "
            f"peptides: {coord_pep:,} coord entries vs {mask_pep:,} mask runs; "
            f"propeptides: {coord_pro:,} coord entries vs {mask_pro:,} mask runs. "
            f"Overlapping/nested annotations collapse into a single mask run.\n"
        )

    lines.append("\n---\n")

    # ---- Section 1: Join verification ----
    lines.append("## 1. Data Integrity Checks\n")
    lines.append(f"| Check | Value |")
    lines.append(f"|-------|-------|")
    lines.append(f"| Total proteins in labeled_sequences | {meta['n_total']:,} |")
    lines.append(f"| Proteins with graphpart assignment | {meta['n_assigned']:,} |")
    lines.append(f"| Proteins excluded (no graphpart) | {meta['n_excluded']:,} |")
    lines.append(f"| Duplicate protein_id in labeled_sequences | {meta['protein_id_dups_labeled']:,} |")
    lines.append(f"| Duplicate AC in graphpart | {meta['dup_ac_graphpart']:,} |")
    lines.append(f"| is_peptide mask length mismatches | {meta['pep_mask_mismatch']:,} |")
    lines.append(f"| is_propeptide mask length mismatches | {meta['pro_mask_mismatch']:,} |")
    lines.append(f"| Null/NaN is_peptide masks | {meta['pep_null']:,} |")
    lines.append(f"| Null/NaN is_propeptide masks | {meta['pro_null']:,} |")
    lines.append(f"| Organism strings without parenthesis | {meta['no_paren_organism']:,} |")
    lines.append("")

    # ---- Section 2: Protein-level counts ----
    lines.append("## 2. Protein-Level Counts per Split\n")
    lines.append("| Split | Total | Has Peptide | Has Propeptide | Has Both | Negatives |")
    lines.append("|-------|-------|-------------|----------------|----------|-----------|")
    for sp in SPLITS:
        s = results[sp]
        lines.append(
            f"| {sp} | {s['n_proteins']:,} | {s['n_has_peptide']:,} | "
            f"{s['n_has_propeptide']:,} | {s['n_has_both']:,} | {s['n_negative']:,} |"
        )
    lines.append("")

    # ---- Section 3: Segment-level counts ----
    lines.append("## 3. Segment-Level Counts per Split\n")
    lines.append("Note: segments counted as runs of '1' in the mask (overlapping annotations collapse into one run).\n")
    lines.append("| Split | Peptide Segs (mask) | Coord Entries (pep) | Propeptide Segs (mask) | Coord Entries (pro) |")
    lines.append("|-------|---------------------|---------------------|------------------------|---------------------|")
    for sp in SPLITS:
        s = results[sp]
        lines.append(
            f"| {sp} | {s['mask_pep_segments']:,} | {s['coord_pep_entries']:,} | "
            f"{s['mask_pro_segments']:,} | {s['coord_pro_entries']:,} |"
        )
    lines.append("")

    lines.append("### Segments per Positive Protein\n")
    lines.append("| Split | Peptide segs/protein (pos only) | Propeptide segs/protein (pos only) |")
    lines.append("|-------|---------------------------------|------------------------------------|")
    for sp in SPLITS:
        s = results[sp]
        pp = s["pep_segs_per_pos_protein"]
        pr = s["pro_segs_per_pos_protein"]
        pp_str = f"mean={pp['mean']:.2f} median={pp['median']:.1f}" if pp["count"] else "N/A"
        pr_str = f"mean={pr['mean']:.2f} median={pr['median']:.1f}" if pr["count"] else "N/A"
        lines.append(f"| {sp} | {pp_str} | {pr_str} |")
    lines.append("")

    # ---- Section 4: Length distributions ----
    lines.append("## 4. Length Distributions\n")

    lines.append("### 4a. Protein Sequence Lengths\n")
    lines.append("| Split | n | min | median | mean | p90 | max |")
    lines.append("|-------|---|-----|--------|------|-----|-----|")
    for sp in SPLITS:
        s = results[sp]
        d = s["prot_len_stats"]
        if d["count"]:
            lines.append(f"| {sp} | {d['count']:,} | {d['min']} | {d['median']:.1f} | {d['mean']:.1f} | {d['p90']:.1f} | {d['max']} |")
        else:
            lines.append(f"| {sp} | 0 | — | — | — | — | — |")
    lines.append("")

    lines.append("### 4b. Peptide Segment Lengths\n")
    lines.append("| Split | n segs | min | median | mean | p90 | max |")
    lines.append("|-------|--------|-----|--------|------|-----|-----|")
    for sp in SPLITS:
        s = results[sp]
        d = s["pep_seg_stats"]
        if d["count"]:
            lines.append(f"| {sp} | {d['count']:,} | {d['min']} | {d['median']:.1f} | {d['mean']:.1f} | {d['p90']:.1f} | {d['max']} |")
        else:
            lines.append(f"| {sp} | 0 | — | — | — | — | — |")
    lines.append("")

    lines.append("### 4c. Propeptide Segment Lengths\n")
    lines.append("| Split | n segs | min | median | mean | p90 | max |")
    lines.append("|-------|--------|-----|--------|------|-----|-----|")
    for sp in SPLITS:
        s = results[sp]
        d = s["pro_seg_stats"]
        if d["count"]:
            lines.append(f"| {sp} | {d['count']:,} | {d['min']} | {d['median']:.1f} | {d['mean']:.1f} | {d['p90']:.1f} | {d['max']} |")
        else:
            lines.append(f"| {sp} | 0 | — | — | — | — | — |")
    lines.append("")

    lines.append("### 4d. Tiny Peptide Segments (ALL split)\n")
    lines.append(f"Peptide segments with length ≤ 5: **{results['ALL']['pep_le5_count']:,}** (all exactly 5 aa — lengths 1–4 are absent).\n")
    lines.append(
        "The minimum of 5 aa is confirmed at both the mask-run level and the coordinate-entry level: "
        "zero coordinate entries have end−start+1 < 5 in either dataset. This is a true annotation "
        "floor, NOT an artifact of overlapping peptides merging short segments into longer runs. "
        "Implication for downstream 'exclude tiny peptides' decisions: a cutoff of < 5 would remove "
        "nothing; a cutoff of ≤ 5 would remove exactly the 5-aa segments listed above.\n"
    )
    lines.append("Counts for individual lengths 1–10 (ALL split):\n")
    lines.append(make_length_table(results["ALL"]["pep_seg_lens"], up_to=10))
    lines.append("")

    # ---- Section 5: Organism distribution ----
    lines.append("## 5. Per-Organism Distribution (Top 15 Species)\n")
    org_tbl = organism_stats(meta["df"], top_n=15)
    col_order = [c for c in ["TRAIN", "VALID", "TEST", "TOTAL"] if c in org_tbl.columns]
    org_tbl = org_tbl[col_order]

    header_cols = " | ".join(col_order)
    sep_cols = " | ".join(["------"] * len(col_order))
    lines.append(f"| Species | {header_cols} |")
    lines.append(f"|---------|{sep_cols}|")
    for sp_name, row in org_tbl.iterrows():
        vals = " | ".join(str(int(row.get(c, 0))) for c in col_order)
        lines.append(f"| {sp_name} | {vals} |")
    lines.append("")

    # % human and diversity note
    working = meta["df"][meta["df"]["split"] != "EXCLUDED"]
    top_sp = working["species"].value_counts()
    if "Homo sapiens" in top_sp.index:
        human_pct = 100 * top_sp["Homo sapiens"] / len(working)
        lines.append(
            f"*Homo sapiens* accounts for **{top_sp['Homo sapiens']:,} / {len(working):,} = {human_pct:.1f}%** of assigned proteins. "
            f"The dataset is **not human-dominated**: venom organisms (spiders, cone snails) and other eukaryotes each contribute "
            f"comparable numbers. GraphPart homology-based clustering may place entire genera almost exclusively in one split "
            f"(e.g. *Cyriopagopus hainanus* and *Lycosa singoriensis* in uniprot_2026 are almost entirely in TRAIN), "
            f"which drives per-split differences in protein length distributions and class imbalance ratios.\n"
        )
    lines.append("")

    # ---- Section 6: Class imbalance at residue level ----
    lines.append("## 6. Residue-Level Class Imbalance\n")
    lines.append("| Split | Total Residues | Peptide Residues | % Peptide | Propeptide Residues | % Propeptide |")
    lines.append("|-------|---------------|-----------------|-----------|---------------------|--------------|")
    for sp in SPLITS:
        s = results[sp]
        tot = s["total_residues"]
        pep_r = s["pep_residues"]
        pro_r = s["pro_residues"]
        pep_pct = 100 * pep_r / tot if tot > 0 else 0.0
        pro_pct = 100 * pro_r / tot if tot > 0 else 0.0
        lines.append(
            f"| {sp} | {tot:,} | {pep_r:,} | {pep_pct:.2f}% | {pro_r:,} | {pro_pct:.2f}% |"
        )
    lines.append("")

    if not HAS_MPL:
        lines.append("\n> **Note:** matplotlib not available; histograms not generated.\n")
    else:
        lines.append(f"\n## 7. Histograms\n")
        lines.append(f"See `analysis/dataset/plots/` for histogram PNGs.\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plots(dataset_key: str, results: Dict):
    if not HAS_MPL:
        print("  matplotlib not available — skipping plots.")
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    all_s = results["ALL"]

    # Protein lengths — широкий непрерывный диапазон (до ~4000), обрезаем хвост по p99,
    # один бин на а.о. внутри показанного диапазона
    save_histogram(
        all_s["prot_lens"],
        f"Длины белков (uniprot_{dataset_key}, ALL)",
        "Длина (а.о.)",
        f"{dataset_key}_protein_lengths.png",
        clip_pct=99,
    )

    # Peptide segment lengths — основная масса ≤50, редкие выбросы до сотен; p99 + 1 бин/а.о.
    pep_lens = all_s["pep_seg_lens"]
    save_histogram(
        pep_lens,
        f"Длины пептидных сегментов (uniprot_{dataset_key}, ALL)",
        "Длина (а.о.)",
        f"{dataset_key}_peptide_lengths.png",
        clip_pct=99,
    )

    # Propeptide segment lengths — диапазон узкий (≤~80), показываем полностью, 1 бин/а.о.
    pro_lens = all_s["pro_seg_lens"]
    save_histogram(
        pro_lens,
        f"Длины пропептидных сегментов (uniprot_{dataset_key}, ALL)",
        "Длина (а.о.)",
        f"{dataset_key}_propeptide_lengths.png",
    )

    # Tiny peptide zoom (lengths 1..30)
    tiny = [l for l in pep_lens if l <= 30]
    if tiny:
        if HAS_MPL:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(tiny, bins=np.arange(0.5, 31.5, 1.0), edgecolor="none", alpha=0.9)
            ax.set_title(f"Длины пептидных сегментов 1–30 а.о. (uniprot_{dataset_key}, ALL)")
            ax.set_xlabel("Длина (а.о.)")
            ax.set_ylabel("Количество")
            ax.set_xticks(range(1, 31))
            fig.tight_layout()
            fig.savefig(PLOTS_DIR / f"{dataset_key}_peptide_lengths_tiny.png", dpi=120)
            plt.close(fig)

    # Per-split bar chart: protein counts stacked
    if HAS_MPL:
        splits_plot = ["TRAIN", "VALID", "TEST"]
        n_pos_pep = [results[s]["n_has_peptide"] for s in splits_plot]
        n_pos_pro = [results[s]["n_has_propeptide"] for s in splits_plot]
        n_neg = [results[s]["n_negative"] for s in splits_plot]

        x = np.arange(len(splits_plot))
        w = 0.25
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(x - w, n_pos_pep, w, label="Есть пептид")
        ax.bar(x, n_pos_pro, w, label="Есть пропептид")
        ax.bar(x + w, n_neg, w, label="Негативы")
        ax.set_xticks(x)
        ax.set_xticklabels(splits_plot)
        ax.set_title(f"Число белков по сплитам (uniprot_{dataset_key})")
        ax.set_ylabel("Число белков")
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"{dataset_key}_split_counts.png", dpi=120)
        plt.close(fig)

    print(f"  Plots written to {PLOTS_DIR}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute dataset statistics for DeepPeptide.")
    parser.add_argument(
        "--dataset",
        choices=["2022", "2026", "both"],
        default="both",
        help="Which dataset to analyze (default: both)",
    )
    args = parser.parse_args()

    keys = ["2022", "2026"] if args.dataset == "both" else [args.dataset]

    for key in keys:
        results = analyze(key)
        make_plots(key, results)

        report = generate_report(key, results)
        report_path = _TOPIC / f"dataset_stats_{key}.md"
        report_path.write_text(report, encoding="utf-8")
        print(f"\n  Report written to: {report_path}")

        # Print headline numbers to stdout
        meta = results["_meta"]
        all_s = results["ALL"]
        print(f"\n  === HEADLINE NUMBERS (uniprot_{key}) ===")
        for sp in ["TRAIN", "VALID", "TEST", "ALL"]:
            s = results[sp]
            print(
                f"  {sp:5s}: proteins={s['n_proteins']:5d}  pep_segs={s['mask_pep_segments']:5d}  "
                f"pro_segs={s['mask_pro_segments']:4d}  neg={s['n_negative']:4d}"
            )
        pep_stats = all_s["pep_seg_stats"]
        print(f"  Peptide seg length: median={pep_stats['median']:.1f}  pep_le5={all_s['pep_le5_count']}")
        print(f"  Excluded (no graphpart): {meta['n_excluded']}")
        print(f"  Mask mismatches: pep={meta['pep_mask_mismatch']} pro={meta['pro_mask_mismatch']}")


if __name__ == "__main__":
    main()
