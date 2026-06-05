#!/usr/bin/env python3
"""
Aho-Corasick-only baseline for DeepPeptide-style peptide/propeptide segmentation.

Intended project layout:

    <repo>/
      src/
        train_loop_aho.py
      data/
        uniprot_2022/
          labeled_sequences.csv
          graphpart_assignments.csv
        aho_train/
          uniprot_2022.tsv
          neuropep.tsv
          ...

Run from repository root, for example:

    python src/train_loop_aho.py \
        --data_file data/uniprot_2022/labeled_sequences.csv \
        --partitioning_file data/uniprot_2022/graphpart_assignments.csv \
        --aho_dir data/aho_train \
        --out_dir runs/aho_only

What this script does:
  1. Loads protein sequences and gold PEPTIDE/PROPEP coordinates from labeled_sequences.csv.
  2. Loads graphpart folds from graphpart_assignments.csv.
  3. Loads normalized Aho dictionaries from data/aho_train/*.tsv or *.csv.
  4. For folded dictionary rows, keeps only rows from the allowed folds.
     For rows/files without fold information, keeps everything as an external source.
  5. Deduplicates dictionary entries by (sequence, label), keeping source metadata.
  6. Scans validation proteins with Aho-Corasick and searches scoring parameters by validation F1.
  7. Rebuilds the dictionary on train+valid folds and predicts the test fold.

The Aho-only model has no embeddings and no neural training. The "training" here is only
validation-based parameter selection for resolving overlapping dictionary hits.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import re
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal

import pandas as pd

CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")
CANONICAL_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")

PROTEIN_ID_COLUMNS = ("protein_id", "AC", "accession", "Entry", "entry", "id")
SEQUENCE_COLUMNS = ("sequence", "Sequence", "protein_sequence", "seq")
PEPTIDE_COORD_COLUMNS = ("coordinates", "peptide_coordinates", "pep_coordinates")
PROPEPTIDE_COORD_COLUMNS = ("propeptide_coordinates", "propep_coordinates")
ORGANISM_COLUMNS = ("organism", "Organism")
GRAPH_ID_COLUMNS = ("AC", "protein_id", "accession", "Entry", "entry", "id")
FOLD_COLUMNS = ("cluster", "fold", "partition")

LABEL_ALIASES = {
    "pep": "pep",
    "peptide": "pep",
    "peptides": "pep",
    "PEPTIDE": "pep",
    "propep": "propep",
    "propeptide": "propep",
    "propeptides": "propep",
    "PROPEP": "propep",
}


class FormatError(ValueError):
    """Raised when an input file does not match an expected DeepPeptide/Aho format."""


@dataclass(frozen=True)
class Segment:
    protein_id: str
    start: int  # 1-based inclusive
    end: int    # 1-based inclusive
    label: Literal["pep", "propep"]
    sequence: str
    score: float = 0.0
    sources: tuple[str, ...] = ()

    @property
    def length(self) -> int:
        return self.end - self.start + 1


@dataclass(frozen=True)
class ProteinRecord:
    protein_id: str
    sequence: str
    fold: int | None
    organism: str | None
    gold: tuple[Segment, ...]


@dataclass
class DictEntry:
    sequence: str
    label: Literal["pep", "propep"]
    sources: Counter[str] = field(default_factory=Counter)
    source_types: dict[str, str] = field(default_factory=dict)  # source -> "folded" | "external"
    folds: set[int] = field(default_factory=set)

    @property
    def n_occurrences(self) -> int:
        return int(sum(self.sources.values()))

    @property
    def n_sources(self) -> int:
        return len(self.sources)


@dataclass(frozen=True)
class RawHit:
    protein_id: str
    start: int
    end: int
    label: Literal["pep", "propep"]
    sequence: str
    sources: tuple[str, ...]
    source_types: tuple[str, ...]
    n_occurrences: int
    n_sources: int

    @property
    def length(self) -> int:
        return self.end - self.start + 1


@dataclass(frozen=True)
class ScoreParams:
    min_len: int
    length_weight: float
    folded_source_weight: float
    external_source_weight: float
    pep_class_weight: float
    propep_class_weight: float
    source_count_weight: float
    occurrence_weight: float


@dataclass(frozen=True)
class DictionaryBundle:
    entries: dict[tuple[str, str], DictEntry]
    entries_by_sequence: dict[str, tuple[DictEntry, ...]]
    patterns: tuple[str, ...]
    source_names: tuple[str, ...]
    n_rows_loaded: int
    n_rows_kept: int

    def sequences_by_label(self) -> dict[str, set[str]]:
        out = {"pep": set(), "propep": set()}
        for (seq, label) in self.entries:
            out[label].add(seq)
        return out


# ----------------------------
# Generic parsing helpers
# ----------------------------


def find_column(df: pd.DataFrame, candidates: Iterable[str], *, required: bool = True) -> str | None:
    columns = list(df.columns)
    for name in candidates:
        if name in df.columns:
            return name
    by_lower = {str(col).lower(): str(col) for col in columns}
    for name in candidates:
        found = by_lower.get(name.lower())
        if found is not None:
            return found
    if required:
        raise FormatError(f"Missing column among {tuple(candidates)!r}. Available columns: {columns!r}")
    return None


def normalize_label(value: Any) -> Literal["pep", "propep"] | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    return LABEL_ALIASES.get(text, LABEL_ALIASES.get(text.lower()))


def clean_sequence(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    seq = str(value).strip().upper()
    seq = re.sub(r"\s+", "", seq)
    seq = seq.replace("-", "")
    return seq


def is_canonical_sequence(seq: str) -> bool:
    return bool(CANONICAL_RE.fullmatch(seq))


def parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_key_float_map(value: str | None) -> dict[str, float]:
    """Parse 'source_a=10,source_b=5' into a mapping. Empty string -> {}."""
    if not value:
        return {}
    result: dict[str, float] = {}
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise FormatError(f"Expected key=value in --source_weights, got {token!r}")
        key, val = token.split("=", 1)
        result[key.strip()] = float(val.strip())
    return result


def parse_coordinate_string(value: Any) -> list[tuple[int, int]]:
    """
    Parse common DeepPeptide/UniProt coordinate encodings into 1-based inclusive intervals.

    Supported examples:
        "23-41,88-101"
        "(23-41),(38-55)"
        "[(23, 41), (88, 101)]"
        "(23, 41)"
    """
    if value is None or pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []

    intervals: list[tuple[int, int]] = []

    # DeepPeptide-style: 23-41,88-101
    for a, b in re.findall(r"(\d+)\s*-\s*(\d+)", text):
        intervals.append((int(a), int(b)))

    if not intervals:
        # Python literal tuple/list style: [(23, 41), (88, 101)]
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            parsed = None

        def add_pair(obj: Any) -> None:
            if isinstance(obj, (tuple, list)) and len(obj) == 2:
                a, b = obj
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    intervals.append((int(a), int(b)))

        if isinstance(parsed, (tuple, list)):
            # One interval: (23, 41)
            if len(parsed) == 2 and all(isinstance(x, (int, float)) for x in parsed):
                add_pair(parsed)
            else:
                for item in parsed:
                    add_pair(item)

    if not intervals:
        # Last-resort parse of "(23, 41), (88, 101)" without requiring a valid literal.
        for a, b in re.findall(r"\(?\s*(\d+)\s*,\s*(\d+)\s*\)?", text):
            intervals.append((int(a), int(b)))

    if not intervals:
        raise FormatError(f"Cannot parse coordinate string: {text!r}")

    clean: list[tuple[int, int]] = []
    for start, end in intervals:
        if start <= 0 or end <= 0:
            raise FormatError(f"Coordinates must be positive 1-based integers: {text!r}")
        if end < start:
            raise FormatError(f"Coordinate end is smaller than start: {text!r}")
        clean.append((start, end))
    return sorted(clean, key=lambda x: (x[0], x[1]))


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    # Let pandas sniff small nonstandard exports.
    with path.open("r", newline="") as f:
        sample = f.read(4096)
    dialect = csv.Sniffer().sniff(sample, delimiters="\t,;")
    return pd.read_csv(path, sep=dialect.delimiter)


def parse_fold_value(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        return int(float(text))
    except ValueError as exc:
        raise FormatError(f"Cannot parse fold value {text!r}") from exc


# ----------------------------
# Protein/gold loading
# ----------------------------


def load_graphpart(path: Path) -> dict[str, int]:
    gp = read_table(path)
    protein_id_col = find_column(gp, GRAPH_ID_COLUMNS)
    fold_col = find_column(gp, FOLD_COLUMNS)
    folds: dict[str, int] = {}
    for _, row in gp.iterrows():
        protein_id = str(row[protein_id_col]).strip()
        fold = parse_fold_value(row[fold_col])
        if protein_id and fold is not None:
            folds[protein_id] = fold
    return folds


def load_proteins(data_file: Path, partitioning_file: Path, *, keep_outside_graphpart: bool = False) -> list[ProteinRecord]:
    df = read_table(data_file)
    folds = load_graphpart(partitioning_file)

    protein_id_col = find_column(df, PROTEIN_ID_COLUMNS)
    sequence_col = find_column(df, SEQUENCE_COLUMNS)
    peptide_coord_col = find_column(df, PEPTIDE_COORD_COLUMNS)
    propeptide_coord_col = find_column(df, PROPEPTIDE_COORD_COLUMNS)
    organism_col = find_column(df, ORGANISM_COLUMNS, required=False)

    records: list[ProteinRecord] = []
    for _, row in df.iterrows():
        protein_id = str(row[protein_id_col]).strip()
        if not protein_id:
            continue
        fold = folds.get(protein_id)
        if fold is None and not keep_outside_graphpart:
            continue

        seq = clean_sequence(row[sequence_col])
        if not seq:
            continue
        organism = str(row[organism_col]).strip() if organism_col is not None and not pd.isna(row[organism_col]) else None

        gold: list[Segment] = []
        for label, coord_col in (("pep", peptide_coord_col), ("propep", propeptide_coord_col)):
            for start, end in parse_coordinate_string(row.get(coord_col, "")):
                if end > len(seq):
                    raise FormatError(
                        f"Gold coordinate outside protein bounds for {protein_id}: {start}-{end}, len={len(seq)}"
                    )
                gold.append(Segment(protein_id, start, end, label, seq[start - 1 : end]))

        records.append(ProteinRecord(protein_id, seq, fold, organism, tuple(sorted(gold, key=lambda x: (x.start, x.end, x.label)))))
    return records


def filter_records_by_partitions(records: list[ProteinRecord], partitions: set[int], *, restrict_ids: set[str] | None = None) -> list[ProteinRecord]:
    out = [r for r in records if r.fold in partitions]
    if restrict_ids is not None:
        out = [r for r in out if r.protein_id in restrict_ids]
    return out


def load_homo_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(path)
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


# ----------------------------
# Aho dictionary loading
# ----------------------------


def iter_aho_files(aho_dir: Path, allowed_sources: set[str] | None = None) -> list[Path]:
    if not aho_dir.exists():
        raise FileNotFoundError(f"Missing Aho dictionary directory: {aho_dir}")
    files = sorted([p for p in aho_dir.iterdir() if p.is_file() and p.suffix.lower() in {".tsv", ".csv"}])
    if allowed_sources is not None:
        files = [p for p in files if p.stem in allowed_sources]
    if not files:
        raise FileNotFoundError(f"No .tsv/.csv dictionary files found in {aho_dir}")
    return files


def load_aho_dictionary(
    aho_dir: Path,
    *,
    allowed_folds: set[int],
    allowed_sources: set[str] | None,
    base_min_len: int,
    max_len: int,
    canonical_only: bool,
    drop_external_sequences: set[str] | None = None,
) -> DictionaryBundle:
    """
    Load all normalized Aho source tables.

    Fold rule:
      - If a row has a non-empty fold value, it is kept only if fold in allowed_folds.
      - If a row has no fold value, it is treated as external and kept entirely.

    This lets data/aho_train/uniprot_2022.tsv be fold-aware, while NeuroPep/DRAMP/etc. can be
    peptide-only external dictionaries without graphpart partitions.
    """
    entries: dict[tuple[str, str], DictEntry] = {}
    source_names: set[str] = set()
    n_rows_loaded = 0
    n_rows_kept = 0
    drop_external_sequences = drop_external_sequences or set()

    for path in iter_aho_files(aho_dir, allowed_sources):
        source = path.stem
        source_names.add(source)
        df = read_table(path)
        seq_col = find_column(df, ("sequence", "Sequence", "seq", "peptide", "peptide_sequence"))
        label_col = find_column(df, ("label", "type", "class", "feature", "kind"))
        fold_col = find_column(df, FOLD_COLUMNS, required=False)

        for _, row in df.iterrows():
            n_rows_loaded += 1
            seq = clean_sequence(row[seq_col])
            label = normalize_label(row[label_col])
            if label is None or not seq:
                continue
            if len(seq) < base_min_len or len(seq) > max_len:
                continue
            if canonical_only and not is_canonical_sequence(seq):
                continue

            row_fold = parse_fold_value(row[fold_col]) if fold_col is not None else None
            is_folded_row = row_fold is not None
            source_type = "folded" if is_folded_row else "external"

            if is_folded_row and row_fold not in allowed_folds:
                continue
            if not is_folded_row and seq in drop_external_sequences:
                continue

            key = (seq, label)
            entry = entries.get(key)
            if entry is None:
                entry = DictEntry(sequence=seq, label=label)
                entries[key] = entry
            entry.sources[source] += 1
            entry.source_types[source] = source_type
            if row_fold is not None:
                entry.folds.add(row_fold)
            n_rows_kept += 1

    entries_by_sequence: dict[str, list[DictEntry]] = defaultdict(list)
    for entry in entries.values():
        entries_by_sequence[entry.sequence].append(entry)

    return DictionaryBundle(
        entries=entries,
        entries_by_sequence={k: tuple(v) for k, v in entries_by_sequence.items()},
        patterns=tuple(sorted(entries_by_sequence)),
        source_names=tuple(sorted(source_names)),
        n_rows_loaded=n_rows_loaded,
        n_rows_kept=n_rows_kept,
    )


# ----------------------------
# Aho-Corasick matcher
# ----------------------------


class PurePythonAho:
    """Minimal dependency-free Aho-Corasick automaton for uppercase amino acid strings."""

    def __init__(self, patterns: Iterable[str]):
        self.next: list[dict[str, int]] = [dict()]
        self.fail: list[int] = [0]
        self.out: list[list[int]] = [[]]
        self.patterns: list[str] = []

        for pattern in patterns:
            if not pattern:
                continue
            self._add(pattern)
        self._build()

    def _add(self, pattern: str) -> None:
        idx = len(self.patterns)
        self.patterns.append(pattern)
        state = 0
        for ch in pattern:
            nxt = self.next[state].get(ch)
            if nxt is None:
                nxt = len(self.next)
                self.next[state][ch] = nxt
                self.next.append({})
                self.fail.append(0)
                self.out.append([])
            state = nxt
        self.out[state].append(idx)

    def _build(self) -> None:
        q: deque[int] = deque()
        for nxt in self.next[0].values():
            q.append(nxt)
            self.fail[nxt] = 0

        while q:
            state = q.popleft()
            for ch, nxt in self.next[state].items():
                q.append(nxt)
                fail_state = self.fail[state]
                while fail_state and ch not in self.next[fail_state]:
                    fail_state = self.fail[fail_state]
                self.fail[nxt] = self.next[fail_state].get(ch, 0)
                self.out[nxt].extend(self.out[self.fail[nxt]])

    def iter(self, text: str) -> Iterator[tuple[int, str]]:
        state = 0
        for i, ch in enumerate(text):
            while state and ch not in self.next[state]:
                state = self.fail[state]
            state = self.next[state].get(ch, 0)
            for pattern_idx in self.out[state]:
                yield i, self.patterns[pattern_idx]


def build_matcher(patterns: Iterable[str]):
    """Use pyahocorasick if installed; otherwise fallback to PurePythonAho."""
    patterns = tuple(patterns)
    try:
        import ahocorasick  # type: ignore
    except Exception:
        return PurePythonAho(patterns)

    automaton = ahocorasick.Automaton()
    for pattern in patterns:
        automaton.add_word(pattern, pattern)
    automaton.make_automaton()
    return automaton


def iter_matches(matcher: Any, sequence: str) -> Iterator[tuple[int, str]]:
    """Yield (end_index_0_based, matched_sequence) for either pyahocorasick or PurePythonAho."""
    for end_idx, value in matcher.iter(sequence):
        # pyahocorasick returns the value we inserted; PurePythonAho returns pattern string.
        yield int(end_idx), str(value)


# ----------------------------
# Prediction/scoring
# ----------------------------


def scan_records(records: list[ProteinRecord], matcher: Any, dictionary: DictionaryBundle) -> dict[str, list[RawHit]]:
    raw_by_protein: dict[str, list[RawHit]] = {}
    for record in records:
        hits: list[RawHit] = []
        for end0, matched_seq in iter_matches(matcher, record.sequence):
            start0 = end0 - len(matched_seq) + 1
            start = start0 + 1
            end = end0 + 1
            for entry in dictionary.entries_by_sequence.get(matched_seq, ()):  # one per label for this sequence
                sources = tuple(sorted(entry.sources))
                source_types = tuple(entry.source_types[s] for s in sources)
                hits.append(
                    RawHit(
                        protein_id=record.protein_id,
                        start=start,
                        end=end,
                        label=entry.label,
                        sequence=matched_seq,
                        sources=sources,
                        source_types=source_types,
                        n_occurrences=entry.n_occurrences,
                        n_sources=entry.n_sources,
                    )
                )
        raw_by_protein[record.protein_id] = hits
    return raw_by_protein


def score_raw_hit(
    hit: RawHit,
    params: ScoreParams,
    *,
    source_weight_overrides: dict[str, float],
    source_agg: Literal["max", "sum"],
) -> float:
    source_scores: list[float] = []
    for source, source_type in zip(hit.sources, hit.source_types):
        if source in source_weight_overrides:
            source_scores.append(source_weight_overrides[source])
        elif source_type == "folded":
            source_scores.append(params.folded_source_weight)
        else:
            source_scores.append(params.external_source_weight)

    if not source_scores:
        source_score = 0.0
    elif source_agg == "sum":
        source_score = float(sum(source_scores))
    else:
        source_score = float(max(source_scores))

    class_score = params.pep_class_weight if hit.label == "pep" else params.propep_class_weight
    return (
        source_score
        + params.length_weight * hit.length
        + class_score
        + params.source_count_weight * math.log1p(hit.n_sources)
        + params.occurrence_weight * math.log1p(hit.n_occurrences)
    )


def select_non_overlapping_hits(
    raw_hits: list[RawHit],
    params: ScoreParams,
    *,
    source_weight_overrides: dict[str, float],
    source_agg: Literal["max", "sum"],
) -> list[Segment]:
    """Resolve overlapping hits by weighted interval scheduling."""
    candidates: list[tuple[RawHit, float]] = []
    for hit in raw_hits:
        if hit.length < params.min_len:
            continue
        score = score_raw_hit(hit, params, source_weight_overrides=source_weight_overrides, source_agg=source_agg)
        if score <= 0:
            continue
        candidates.append((hit, score))

    if not candidates:
        return []

    # Sort by end, then start. DP uses p[j] = rightmost non-overlapping interval before j.
    candidates.sort(key=lambda x: (x[0].end, x[0].start, -x[1], x[0].label))
    ends = [h.end for h, _ in candidates]

    def rightmost_non_overlap(j: int) -> int:
        lo, hi = 0, j - 1
        ans = -1
        start_j = candidates[j][0].start
        while lo <= hi:
            mid = (lo + hi) // 2
            if ends[mid] < start_j:
                ans = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return ans

    n = len(candidates)
    p = [rightmost_non_overlap(j) for j in range(n)]
    dp = [0.0] * (n + 1)
    take = [False] * n

    for j in range(1, n + 1):
        hit, score = candidates[j - 1]
        take_score = score + dp[p[j - 1] + 1]
        skip_score = dp[j - 1]
        if take_score > skip_score + 1e-12:
            dp[j] = take_score
            take[j - 1] = True
        else:
            dp[j] = skip_score

    selected: list[Segment] = []
    j = n
    while j > 0:
        if take[j - 1] and (candidates[j - 1][1] + dp[p[j - 1] + 1]) >= dp[j] - 1e-12:
            hit, score = candidates[j - 1]
            selected.append(
                Segment(
                    protein_id=hit.protein_id,
                    start=hit.start,
                    end=hit.end,
                    label=hit.label,
                    sequence=hit.sequence,
                    score=score,
                    sources=hit.sources,
                )
            )
            j = p[j - 1] + 1
        else:
            j -= 1

    return sorted(selected, key=lambda x: (x.start, x.end, x.label))


def predict_from_raw_hits(
    records: list[ProteinRecord],
    raw_hits_by_protein: dict[str, list[RawHit]],
    params: ScoreParams,
    *,
    source_weight_overrides: dict[str, float],
    source_agg: Literal["max", "sum"],
) -> dict[str, list[Segment]]:
    out: dict[str, list[Segment]] = {}
    for record in records:
        out[record.protein_id] = select_non_overlapping_hits(
            raw_hits_by_protein.get(record.protein_id, []),
            params,
            source_weight_overrides=source_weight_overrides,
            source_agg=source_agg,
        )
    return out


# ----------------------------
# Metrics
# ----------------------------


def match_segments(
    gold: list[Segment],
    pred: list[Segment],
    *,
    tolerance: int,
    label: str | None = None,
) -> tuple[int, int, int]:
    """One-to-one segment matching with same protein, boundary tolerance, and same label."""
    if label is not None:
        gold = [s for s in gold if s.label == label]
        pred = [s for s in pred if s.label == label]

    unmatched_gold = set(range(len(gold)))
    tp = 0
    fp = 0

    # High-score predictions first; stable by interval afterwards.
    for pseg in sorted(pred, key=lambda s: (-s.score, s.start, s.end, s.label)):
        best_idx = None
        best_dist = None
        for gi in unmatched_gold:
            gseg = gold[gi]
            if gseg.protein_id != pseg.protein_id:
                continue
            if gseg.label != pseg.label:
                continue
            ds = abs(gseg.start - pseg.start)
            de = abs(gseg.end - pseg.end)
            if ds <= tolerance and de <= tolerance:
                dist = ds + de
                if best_dist is None or dist < best_dist:
                    best_idx = gi
                    best_dist = dist
        if best_idx is None:
            fp += 1
        else:
            tp += 1
            unmatched_gold.remove(best_idx)

    fn = len(unmatched_gold)
    return tp, fp, fn


def prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def flatten_gold(records: list[ProteinRecord]) -> list[Segment]:
    return [seg for rec in records for seg in rec.gold]


def flatten_pred(predictions: dict[str, list[Segment]]) -> list[Segment]:
    return [seg for segs in predictions.values() for seg in segs]


def compute_metrics(
    records: list[ProteinRecord],
    predictions: dict[str, list[Segment]],
    *,
    raw_hits_by_protein: dict[str, list[RawHit]] | None,
    dictionary_sequences_by_label: dict[str, set[str]] | None,
    tolerance: int,
) -> dict[str, Any]:
    gold_all = flatten_gold(records)
    pred_all = flatten_pred(predictions)
    metrics: dict[str, Any] = {
        "n_proteins": len(records),
        "n_gold_segments": len(gold_all),
        "n_pred_segments": len(pred_all),
        "tolerance": tolerance,
    }

    for name, label in (("all", None), ("pep", "pep"), ("propep", "propep")):
        tp, fp, fn = match_segments(gold_all, pred_all, tolerance=tolerance, label=label)
        for k, v in prf(tp, fp, fn).items():
            metrics[f"segment/{name}/{k}"] = v

    metrics["stopping_metric"] = (metrics["segment/pep/f1"] + metrics["segment/propep/f1"]) / 2

    # Known/novel recall by exact sequence membership in the actual dictionary used for prediction.
    if dictionary_sequences_by_label is not None:
        for name, require_known in (("known", True), ("novel", False)):
            subset_gold = []
            for seg in gold_all:
                is_known = seg.sequence in dictionary_sequences_by_label.get(seg.label, set())
                if is_known == require_known:
                    subset_gold.append(seg)
            tp, _, fn = match_segments(subset_gold, pred_all, tolerance=tolerance, label=None)
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            metrics[f"gold_{name}/n"] = len(subset_gold)
            metrics[f"gold_{name}/matched"] = tp
            metrics[f"gold_{name}/recall"] = recall

    # Proteins with no raw Aho hits. Selected predictions are necessarily empty, but keep the metric explicit.
    if raw_hits_by_protein is not None:
        no_hit_ids = {rec.protein_id for rec in records if not raw_hits_by_protein.get(rec.protein_id)}
        metrics["no_hit_proteins/n"] = len(no_hit_ids)
        if no_hit_ids:
            no_hit_records = [rec for rec in records if rec.protein_id in no_hit_ids]
            no_hit_gold = flatten_gold(no_hit_records)
            no_hit_pred = [seg for pid, segs in predictions.items() if pid in no_hit_ids for seg in segs]
            tp, fp, fn = match_segments(no_hit_gold, no_hit_pred, tolerance=tolerance, label=None)
            for k, v in prf(tp, fp, fn).items():
                metrics[f"no_hit_proteins/segment/all/{k}"] = v

    return metrics


# ----------------------------
# Grid search and output
# ----------------------------


def make_param_grid(args: argparse.Namespace) -> list[ScoreParams]:
    grid: list[ScoreParams] = []
    for min_len in parse_int_list(args.grid_min_len):
        for length_weight in parse_float_list(args.grid_length_weight):
            for folded_source_weight in parse_float_list(args.grid_folded_source_weight):
                for external_source_weight in parse_float_list(args.grid_external_source_weight):
                    for pep_class_weight in parse_float_list(args.grid_pep_class_weight):
                        for propep_class_weight in parse_float_list(args.grid_propep_class_weight):
                            for source_count_weight in parse_float_list(args.grid_source_count_weight):
                                for occurrence_weight in parse_float_list(args.grid_occurrence_weight):
                                    grid.append(
                                        ScoreParams(
                                            min_len=min_len,
                                            length_weight=length_weight,
                                            folded_source_weight=folded_source_weight,
                                            external_source_weight=external_source_weight,
                                            pep_class_weight=pep_class_weight,
                                            propep_class_weight=propep_class_weight,
                                            source_count_weight=source_count_weight,
                                            occurrence_weight=occurrence_weight,
                                        )
                                    )
    if not grid:
        raise ValueError("Empty parameter grid")
    return grid


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def write_metrics_log(path: Path, prefix: str, metrics: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(f"[{prefix}]\n")
        for key in sorted(metrics):
            f.write(f"{key}: {metrics[key]}\n")
        f.write("\n")


def write_predictions(path: Path, predictions: dict[str, list[Segment]]) -> None:
    rows: list[dict[str, Any]] = []
    for protein_id, segs in predictions.items():
        for seg in segs:
            rows.append(
                {
                    "protein_id": protein_id,
                    "start": seg.start,
                    "end": seg.end,
                    "label": seg.label,
                    "sequence": seg.sequence,
                    "length": seg.length,
                    "score": seg.score,
                    "sources": ";".join(seg.sources),
                }
            )
    pd.DataFrame(rows, columns=["protein_id", "start", "end", "label", "sequence", "length", "score", "sources"]).to_csv(
        path, sep="\t", index=False
    )


def write_gold(path: Path, records: list[ProteinRecord]) -> None:
    rows: list[dict[str, Any]] = []
    for rec in records:
        for seg in rec.gold:
            rows.append(
                {
                    "protein_id": rec.protein_id,
                    "fold": rec.fold,
                    "start": seg.start,
                    "end": seg.end,
                    "label": seg.label,
                    "sequence": seg.sequence,
                    "length": seg.length,
                }
            )
    pd.DataFrame(rows, columns=["protein_id", "fold", "start", "end", "label", "sequence", "length"]).to_csv(
        path, sep="\t", index=False
    )


def evaluation_gold_sequences(records: list[ProteinRecord]) -> set[str]:
    return {seg.sequence for rec in records for seg in rec.gold}


def prepare_dictionary_and_scan(
    *,
    args: argparse.Namespace,
    allowed_folds: set[int],
    eval_records: list[ProteinRecord],
    drop_eval_gold_from_external: bool,
) -> tuple[DictionaryBundle, dict[str, list[RawHit]]]:
    drop_external = evaluation_gold_sequences(eval_records) if drop_eval_gold_from_external else set()
    dictionary = load_aho_dictionary(
        Path(args.aho_dir),
        allowed_folds=allowed_folds,
        allowed_sources=set(args.sources.split(",")) if args.sources else None,
        base_min_len=min(parse_int_list(args.grid_min_len)),
        max_len=args.max_len,
        canonical_only=not args.allow_noncanonical,
        drop_external_sequences=drop_external,
    )
    matcher = build_matcher(dictionary.patterns)
    raw_hits = scan_records(eval_records, matcher, dictionary)
    return dictionary, raw_hits


def train(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "config.json", vars(args))

    train_partitions = set(parse_int_list(args.train_partitions))
    valid_partitions = set(parse_int_list(args.valid_partitions))
    test_partitions = set(parse_int_list(args.test_partitions))

    homo_ids = load_homo_ids(Path(args.homo_ids_file)) if args.homo_only else None

    all_records = load_proteins(
        Path(args.data_file),
        Path(args.partitioning_file),
        keep_outside_graphpart=args.keep_outside_graphpart,
    )
    train_records = filter_records_by_partitions(all_records, train_partitions, restrict_ids=homo_ids)
    valid_records = filter_records_by_partitions(all_records, valid_partitions, restrict_ids=homo_ids)
    test_records = filter_records_by_partitions(all_records, test_partitions, restrict_ids=homo_ids)

    print(
        f"Loaded proteins: {len(train_records)} train {sorted(train_partitions)}, "
        f"{len(valid_records)} valid {sorted(valid_partitions)}, {len(test_records)} test {sorted(test_partitions)}"
    )

    source_weight_overrides = parse_key_float_map(args.source_weights)
    source_agg: Literal["max", "sum"] = args.source_agg
    drop_eval_gold_from_external = args.external_gold_policy == "drop_eval_exact"

    # Validation dictionary: internal/folded sources from train folds only; external sources whole-file.
    valid_dictionary, valid_raw_hits = prepare_dictionary_and_scan(
        args=args,
        allowed_folds=train_partitions,
        eval_records=valid_records,
        drop_eval_gold_from_external=drop_eval_gold_from_external,
    )
    print(
        f"Validation dictionary: {len(valid_dictionary.entries)} dedup entries, "
        f"{len(valid_dictionary.patterns)} unique patterns, "
        f"rows kept/loaded={valid_dictionary.n_rows_kept}/{valid_dictionary.n_rows_loaded}, "
        f"sources={valid_dictionary.source_names}"
    )

    param_grid = make_param_grid(args)
    grid_rows: list[dict[str, Any]] = []
    best_params: ScoreParams | None = None
    best_metrics: dict[str, Any] | None = None
    best_predictions: dict[str, list[Segment]] | None = None

    dict_seqs_by_label = valid_dictionary.sequences_by_label()
    for idx, params in enumerate(param_grid, start=1):
        predictions = predict_from_raw_hits(
            valid_records,
            valid_raw_hits,
            params,
            source_weight_overrides=source_weight_overrides,
            source_agg=source_agg,
        )
        metrics = compute_metrics(
            valid_records,
            predictions,
            raw_hits_by_protein=valid_raw_hits,
            dictionary_sequences_by_label=dict_seqs_by_label,
            tolerance=args.tolerance,
        )
        row = {**asdict(params), **metrics}
        grid_rows.append(row)

        if best_metrics is None or metrics[args.selection_metric] > best_metrics[args.selection_metric]:
            best_metrics = metrics
            best_params = params
            best_predictions = predictions

        if args.verbose and (idx == 1 or idx == len(param_grid) or idx % max(1, len(param_grid) // 10) == 0):
            print(f"Grid {idx}/{len(param_grid)}: best {args.selection_metric}={best_metrics[args.selection_metric]:.4f}")

    assert best_params is not None and best_metrics is not None and best_predictions is not None

    grid_df = pd.DataFrame(grid_rows).sort_values(args.selection_metric, ascending=False)
    grid_df.to_csv(out_dir / "valid_grid.tsv", sep="\t", index=False)
    write_json(out_dir / "best_params.json", asdict(best_params))
    write_json(out_dir / "valid_metrics.json", best_metrics)
    write_predictions(out_dir / "valid_predictions.tsv", best_predictions)
    write_gold(out_dir / "valid_gold.tsv", valid_records)
    write_metrics_log(out_dir / "all_metrics.txt", "Valid", best_metrics)

    print(f"Best validation params: {asdict(best_params)}")
    print(f"Best validation {args.selection_metric}: {best_metrics[args.selection_metric]:.4f}")

    # Final test dictionary: by default use train+valid folds for folded/internal sources.
    final_allowed_folds = set(train_partitions)
    if args.include_valid_in_final_dictionary:
        final_allowed_folds |= set(valid_partitions)

    test_dictionary, test_raw_hits = prepare_dictionary_and_scan(
        args=args,
        allowed_folds=final_allowed_folds,
        eval_records=test_records,
        drop_eval_gold_from_external=drop_eval_gold_from_external,
    )
    print(
        f"Test dictionary: {len(test_dictionary.entries)} dedup entries, "
        f"{len(test_dictionary.patterns)} unique patterns, "
        f"rows kept/loaded={test_dictionary.n_rows_kept}/{test_dictionary.n_rows_loaded}, "
        f"sources={test_dictionary.source_names}"
    )

    test_predictions = predict_from_raw_hits(
        test_records,
        test_raw_hits,
        best_params,
        source_weight_overrides=source_weight_overrides,
        source_agg=source_agg,
    )
    test_metrics = compute_metrics(
        test_records,
        test_predictions,
        raw_hits_by_protein=test_raw_hits,
        dictionary_sequences_by_label=test_dictionary.sequences_by_label(),
        tolerance=args.tolerance,
    )

    write_json(out_dir / "test_metrics.json", test_metrics)
    write_predictions(out_dir / "test_predictions.tsv", test_predictions)
    write_gold(out_dir / "test_gold.tsv", test_records)
    write_metrics_log(out_dir / "all_metrics.txt", "Test", test_metrics)

    # Save compact run metadata for reproducibility.
    write_json(
        out_dir / "dictionary_summary.json",
        {
            "validation": {
                "allowed_folds": sorted(train_partitions),
                "n_entries": len(valid_dictionary.entries),
                "n_patterns": len(valid_dictionary.patterns),
                "n_rows_loaded": valid_dictionary.n_rows_loaded,
                "n_rows_kept": valid_dictionary.n_rows_kept,
                "sources": valid_dictionary.source_names,
            },
            "test": {
                "allowed_folds": sorted(final_allowed_folds),
                "n_entries": len(test_dictionary.entries),
                "n_patterns": len(test_dictionary.patterns),
                "n_rows_loaded": test_dictionary.n_rows_loaded,
                "n_rows_kept": test_dictionary.n_rows_kept,
                "sources": test_dictionary.source_names,
            },
            "external_gold_policy": args.external_gold_policy,
            "source_weight_overrides": source_weight_overrides,
        },
    )

    print(f"Test {args.selection_metric}: {test_metrics.get(args.selection_metric, float('nan')):.4f}")
    return best_metrics, test_metrics


# ----------------------------
# CLI
# ----------------------------


def parse_arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aho-only baseline train loop for peptide/propeptide segmentation.")

    p.add_argument("--data_file", "-df", type=str, default="data/uniprot_2022/labeled_sequences.csv")
    p.add_argument("--partitioning_file", "-pf", type=str, default="data/uniprot_2022/graphpart_assignments.csv")
    p.add_argument("--aho_dir", type=str, default="data/aho_train")
    p.add_argument("--out_dir", "-od", type=str, default="runs/aho_only")

    p.add_argument("--train_partitions", type=str, default="0,1,2")
    p.add_argument("--valid_partitions", type=str, default="3")
    p.add_argument("--test_partitions", type=str, default="4")
    p.add_argument("--include_valid_in_final_dictionary", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--sources", type=str, default="", help="Optional comma-separated source stems from data/aho_train to include.")
    p.add_argument(
        "--external_gold_policy",
        choices=["keep", "drop_eval_exact"],
        default="keep",
        help="keep: use external dictionaries as known scientific knowledge. drop_eval_exact: remove eval gold exact sequences from external rows only, useful as a strict leakage sensitivity check.",
    )

    p.add_argument("--min_len", type=int, default=5, help="Legacy alias; not used if --grid_min_len is set.")
    p.add_argument("--max_len", type=int, default=50)
    p.add_argument("--allow_noncanonical", action="store_true")
    p.add_argument("--keep_outside_graphpart", action="store_true")

    # Validation grid. Defaults are intentionally small to keep the first run cheap.
    p.add_argument("--grid_min_len", type=str, default="5,8,10")
    p.add_argument("--grid_length_weight", type=str, default="0,0.25,1.0")
    p.add_argument("--grid_folded_source_weight", type=str, default="100")
    p.add_argument("--grid_external_source_weight", type=str, default="20,50,100")
    p.add_argument("--grid_pep_class_weight", type=str, default="0")
    p.add_argument("--grid_propep_class_weight", type=str, default="0,10")
    p.add_argument("--grid_source_count_weight", type=str, default="0")
    p.add_argument("--grid_occurrence_weight", type=str, default="0")

    p.add_argument(
        "--source_weights",
        type=str,
        default="",
        help="Optional fixed source overrides, e.g. 'uniprot_2022=100,neuropep=80'. Overrides folded/external grid weights for listed sources.",
    )
    p.add_argument("--source_agg", choices=["max", "sum"], default="max")

    p.add_argument("--tolerance", type=int, default=3)
    p.add_argument(
        "--selection_metric",
        type=str,
        default="stopping_metric",
        help="Metric key used to choose validation params. Default averages pep/propep segment F1.",
    )

    p.add_argument("--homo_only", action="store_true")
    p.add_argument("--homo_ids_file", type=str, default="data/protein_id_homo.txt")
    p.add_argument("--verbose", action="store_true")

    args = p.parse_args()

    # Backward-friendly behavior: if user passes --min_len but leaves grid default, prepend it.
    if args.min_len != 5 and args.grid_min_len == "5,8,10":
        args.grid_min_len = str(args.min_len)

    os.makedirs(args.out_dir, exist_ok=True)
    return args


if __name__ == "__main__":
    train(parse_arguments())
