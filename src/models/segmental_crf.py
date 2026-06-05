"""Relative-position segmental CRF for PEPTIDE/PROPEP span decoding.

This module is intentionally independent from ``multi_tag_crf.py``.  The old CRF
scores residue-level latent states; this module scores labeled spans directly
(semi-Markov / segmental CRF).  It keeps the public return format used by the
existing training loop: ``(probs, viterbi_paths, loss)`` when targets are given.

Label convention:
    0 = None/background
    1 = PEPTIDE
    2 = PROPEP, only when num_labels == 3

For metric compatibility, decoded spans are converted back to pseudo multistate
paths with the old state IDs:
    None -> 0
    PEPTIDE -> 1 ... 50
    PROPEP -> 51 ... 100
The metrics only need the start/end state IDs, so the internal pseudo path does
not have to be a legal path under the old CRF grammar.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


NONE_LABEL = 0
PEPTIDE_LABEL = 1
PROPEPTIDE_LABEL = 2
START_LABEL = -1
OLD_STATE_MAX_LEN = 50
OLD_NUM_STATES_WITH_PROPEP = 101
OLD_NUM_STATES_PEPTIDE_ONLY = 51


def _logaddexp_optional(a: Optional[torch.Tensor], b: torch.Tensor) -> torch.Tensor:
    if a is None:
        return b
    return torch.logaddexp(a, b)


class RelativePositionSegmentalCRF(nn.Module):
    """Length-aware segmental CRF with relative-position phase scoring.

    Args:
        max_seg_len: Maximum length of PEPTIDE/PROPEP spans.  The current
            DeepPeptide-style CRF uses 50 latent states per segment class, so 50
            is the natural default.
        min_seg_len: Minimum allowed PEPTIDE/PROPEP length.  Set to 5 to mimic
            the old CRF grammar; set to 1 for a more general segmental CRF.
        phase_basis_count: Number of smooth phase bases on r in [0, 1].
        phase_basis_sigma: Gaussian/RBF width for relative-position bases.
        relative_score_scale: Multiplier for the relative-position score inside
            PEPTIDE/PROPEP span scores.
        length_prior_scale: Multiplier for learned length priors.
        transition_scale: Multiplier for learned segment-label transition scores.
        allow_same_label_segments: If false, PEPTIDE->PEPTIDE and
            PROPEP->PROPEP span transitions are disallowed.  In this dataset
            adjacent same-label annotations can exist, so the recommended/default
            value is true.
        num_labels: 2 for None/PEPTIDE, 3 for None/PEPTIDE/PROPEP.
    """

    def __init__(
        self,
        max_seg_len: int = 50,
        min_seg_len: int = 5,
        phase_basis_count: int = 5,
        phase_basis_sigma: float = 0.20,
        relative_score_scale: float = 0.5,
        length_prior_scale: float = 1.0,
        transition_scale: float = 1.0,
        allow_same_label_segments: bool = True,
        num_labels: int = 3,
    ) -> None:
        super().__init__()
        if num_labels not in (2, 3):
            raise ValueError(f"num_labels must be 2 or 3, got {num_labels}")
        if max_seg_len < 1:
            raise ValueError(f"max_seg_len must be >= 1, got {max_seg_len}")
        if min_seg_len < 1 or min_seg_len > max_seg_len:
            raise ValueError(
                f"min_seg_len must be in [1, max_seg_len], got {min_seg_len} with max_seg_len={max_seg_len}"
            )
        if phase_basis_count < 1:
            raise ValueError(f"phase_basis_count must be >= 1, got {phase_basis_count}")
        if phase_basis_sigma <= 0:
            raise ValueError(f"phase_basis_sigma must be > 0, got {phase_basis_sigma}")

        self.max_seg_len = int(max_seg_len)
        self.min_seg_len = int(min_seg_len)
        self.phase_basis_count = int(phase_basis_count)
        self.phase_basis_sigma = float(phase_basis_sigma)
        self.relative_score_scale = float(relative_score_scale)
        self.length_prior_scale = float(length_prior_scale)
        self.transition_scale = float(transition_scale)
        self.allow_same_label_segments = bool(allow_same_label_segments)
        self.num_labels = int(num_labels)
        self.num_segment_labels = self.num_labels - 1  # PEPTIDE, optionally PROPEP
        self.num_states = OLD_NUM_STATES_WITH_PROPEP if self.num_labels == 3 else OLD_NUM_STATES_PEPTIDE_ONLY

        # Learned priors for PEPTIDE/PROPEP lengths. Index 0 is unused.
        self.length_prior = nn.Parameter(torch.zeros(self.num_segment_labels, self.max_seg_len + 1))

        # Segment-level transitions.  START transitions are separate.  The masks
        # are applied in code by refusing disallowed transitions.
        self.start_transitions = nn.Parameter(torch.zeros(self.num_labels))
        self.transitions = nn.Parameter(torch.zeros(self.num_labels, self.num_labels))
        self.end_transitions = nn.Parameter(torch.zeros(self.num_labels))

        basis = self._build_basis_by_length(self.max_seg_len, self.phase_basis_count, self.phase_basis_sigma)
        self.register_buffer("basis_by_length", basis, persistent=False)

    @staticmethod
    def _build_basis_by_length(max_seg_len: int, basis_count: int, sigma: float) -> torch.Tensor:
        """Return tensor [max_seg_len + 1, max_seg_len, K]."""
        centers = torch.linspace(0.0, 1.0, basis_count)
        out = torch.zeros(max_seg_len + 1, max_seg_len, basis_count)
        for m in range(1, max_seg_len + 1):
            if m == 1:
                r = torch.tensor([0.5])
            else:
                r = torch.linspace(0.0, 1.0, m)
            vals = torch.exp(-((r[:, None] - centers[None, :]) ** 2) / (2.0 * sigma * sigma))
            vals = vals / vals.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            out[m, :m, :] = vals
        return out

    def _is_start_state_for_label(self, state: int, label: int) -> bool:
        # Gold labels produced by dataset.py use the old fixed multistate coding:
        # PEPTIDE start = 1, PEPTIDE end = 50, PROPEP start = 51, PROPEP end = 100.
        # This must stay fixed even if max_seg_len for the segmental CRF is changed.
        if label == PEPTIDE_LABEL:
            return state == 1
        if label == PROPEPTIDE_LABEL:
            return state == OLD_STATE_MAX_LEN + 1
        return False

    def _state_to_label(self, state: int) -> int:
        # Interpret existing dataset labels, not the new segmental max_seg_len.
        if state == 0:
            return NONE_LABEL
        if 1 <= state <= OLD_STATE_MAX_LEN:
            return PEPTIDE_LABEL
        if self.num_labels == 3 and (OLD_STATE_MAX_LEN + 1) <= state <= (2 * OLD_STATE_MAX_LEN):
            return PROPEPTIDE_LABEL
        # Padding and unexpected states are treated as background; masked positions
        # are cut away before this is called.
        return NONE_LABEL

    def _targets_to_spans(self, states: torch.Tensor, length: int) -> List[Tuple[int, int, int]]:
        """Convert old multistate target path to full-covering coarse spans.

        Returns a list of (start, end, label), 0-based inclusive.  Explicit
        start states 1/51 always start a new annotated segment, even when the
        previous segment has the same coarse label.  Without this, two adjacent
        peptides can be incorrectly merged into a span longer than max_seg_len.
        """
        if length <= 0:
            return []
        state_list = [int(x) for x in states[:length].detach().cpu().tolist()]
        labels = [self._state_to_label(x) for x in state_list]

        spans: List[Tuple[int, int, int]] = []
        start = 0
        cur_label = labels[0]
        for pos in range(1, length):
            lab = labels[pos]
            starts_same_label_segment = (
                lab == cur_label
                and lab != NONE_LABEL
                and self._is_start_state_for_label(state_list[pos], lab)
            )
            if lab != cur_label or starts_same_label_segment:
                spans.append((start, pos - 1, cur_label))
                start = pos
                cur_label = lab
        spans.append((start, length - 1, cur_label))
        return spans

    def _transition_allowed(self, prev_label: int, curr_label: int) -> bool:
        if curr_label < 0 or curr_label >= self.num_labels:
            return False
        if prev_label == START_LABEL:
            return True
        if prev_label < 0 or prev_label >= self.num_labels:
            return False
        if prev_label == NONE_LABEL and curr_label == NONE_LABEL:
            return True
        if prev_label == curr_label and curr_label != NONE_LABEL:
            return self.allow_same_label_segments
        return True

    def _transition_score(
        self,
        prev_label: int,
        curr_label: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if not self._transition_allowed(prev_label, curr_label):
            return None
        if prev_label == START_LABEL:
            return self.transition_scale * self.start_transitions[curr_label].to(dtype=dtype, device=device)
        if prev_label == NONE_LABEL and curr_label == NONE_LABEL:
            return torch.zeros((), dtype=dtype, device=device)
        return self.transition_scale * self.transitions[prev_label, curr_label].to(dtype=dtype, device=device)

    def _build_span_cache(
        self,
        coarse_logits: torch.Tensor,
        phase_logits: torch.Tensor,
        length: int,
    ):
        """Precompute differentiable span scores for one sequence.

        This optimized version keeps exactly the same scoring as the initial
        prototype, but stores span scores by length as dense tensors instead of
        a nested Python list of one tensor per (label, start, length).  For each
        candidate length m, relative-position scores are computed with a single
        grouped dot product via conv1d:

            rel_score[start, m] = sum_j sum_k phase[start+j, k] * basis[m, j, k]

        Returns:
            prefix: [length + 1, num_labels]
            none_unit_scores: [length]
            segment_scores[label][m]: [length - m + 1] or None
        """
        coarse = coarse_logits[:length]
        phase = phase_logits[:length]
        dtype = coarse.dtype
        device = coarse.device

        zero_row = coarse.new_zeros(1, self.num_labels)
        prefix = torch.cat([zero_row, coarse.cumsum(dim=0)], dim=0)
        none_unit_scores = coarse[:, NONE_LABEL]

        # segment_scores[0] is unused because background is handled one residue
        # at a time in the DP.  For PEPTIDE/PROPEP, segment_scores[label][m]
        # contains all span scores of length m for starts 0..length-m.
        segment_scores: List[Optional[List[Optional[torch.Tensor]]]] = [None for _ in range(self.num_labels)]

        for label in range(1, self.num_labels):
            phase_label_idx = label - 1
            scores_by_len: List[Optional[torch.Tensor]] = [None for _ in range(self.max_seg_len + 1)]

            # [L, K] -> [1, K, L] for conv1d.  conv1d with weight [1, K, m]
            # computes the desired sliding weighted sum over residue phase logits.
            phase_input = phase[:, phase_label_idx, :].transpose(0, 1).unsqueeze(0)

            max_m = min(self.max_seg_len, length)
            for m in range(self.min_seg_len, max_m + 1):
                n_spans = length - m + 1
                if n_spans <= 0:
                    continue

                coarse_sums = prefix[m : length + 1, label] - prefix[:n_spans, label]
                basis = self.basis_by_length[m, :m, :].to(dtype=dtype, device=device)
                weight = basis.transpose(0, 1).unsqueeze(0).contiguous()  # [1, K, m]
                rel_scores = F.conv1d(phase_input, weight=weight).view(n_spans)
                len_score = self.length_prior_scale * self.length_prior[phase_label_idx, m].to(dtype=dtype, device=device)
                scores_by_len[m] = coarse_sums + self.relative_score_scale * rel_scores + len_score

            segment_scores[label] = scores_by_len

        return prefix, none_unit_scores, segment_scores

    def _span_score_from_cache(
        self,
        prefix: torch.Tensor,
        none_unit_scores: torch.Tensor,
        segment_scores,
        start: int,
        end: int,
        label: int,
    ) -> torch.Tensor:
        if label == NONE_LABEL:
            return prefix[end + 1, NONE_LABEL] - prefix[start, NONE_LABEL]
        m = end - start + 1
        if label != NONE_LABEL and (m > self.max_seg_len or m < self.min_seg_len):
            raise ValueError(
                f"Invalid gold segment label={label}, start={start}, end={end}, length={m}; "
                f"allowed non-background lengths are {self.min_seg_len}..{self.max_seg_len}. "
                f"If the raw annotation has no segment this long, check adjacent same-label handling."
            )
        if label < 0 or label >= len(segment_scores) or segment_scores[label] is None:
            raise ValueError(f"Invalid gold segment label={label}.")
        scores_for_m = segment_scores[label][m]
        if scores_for_m is None or start < 0 or start >= scores_for_m.numel():
            raise ValueError(
                f"Invalid gold segment label={label}, start={start}, end={end}, length={m}; "
                f"allowed non-background lengths are {self.min_seg_len}..{self.max_seg_len}."
            )
        return scores_for_m[start]

    def _log_partition_single_from_cache(
        self,
        prefix: torch.Tensor,
        none_unit_scores: torch.Tensor,
        segment_scores,
        length: int,
    ) -> torch.Tensor:
        dtype = prefix.dtype
        device = prefix.device
        zero = torch.zeros((), dtype=dtype, device=device)

        dp: List[List[Optional[torch.Tensor]]] = [[None for _ in range(self.num_labels)] for _ in range(length + 1)]

        for pos in range(length + 1):
            prev_candidates: List[Tuple[int, torch.Tensor]] = []
            if pos == 0:
                prev_candidates.append((START_LABEL, zero))
            for lab in range(self.num_labels):
                if dp[pos][lab] is not None:
                    prev_candidates.append((lab, dp[pos][lab]))
            if not prev_candidates:
                continue

            # Add one background residue.
            if pos < length:
                curr = NONE_LABEL
                span_score = none_unit_scores[pos]
                for prev_lab, prev_score in prev_candidates:
                    tr = self._transition_score(prev_lab, curr, dtype, device)
                    if tr is None:
                        continue
                    cand = prev_score + tr + span_score
                    dp[pos + 1][curr] = _logaddexp_optional(dp[pos + 1][curr], cand)

            # Add PEPTIDE / PROPEP segment jumps.
            max_m = min(self.max_seg_len, length - pos)
            if max_m < self.min_seg_len:
                continue
            for curr in range(1, self.num_labels):
                curr_scores_by_len = segment_scores[curr]
                if curr_scores_by_len is None:
                    continue
                for m in range(self.min_seg_len, max_m + 1):
                    scores_for_m = curr_scores_by_len[m]
                    if scores_for_m is None or pos >= scores_for_m.numel():
                        continue
                    span_score = scores_for_m[pos]
                    new_pos = pos + m
                    for prev_lab, prev_score in prev_candidates:
                        tr = self._transition_score(prev_lab, curr, dtype, device)
                        if tr is None:
                            continue
                        cand = prev_score + tr + span_score
                        dp[new_pos][curr] = _logaddexp_optional(dp[new_pos][curr], cand)

        finals = []
        for lab in range(self.num_labels):
            if dp[length][lab] is not None:
                finals.append(dp[length][lab] + self.transition_scale * self.end_transitions[lab].to(dtype=dtype, device=device))
        if not finals:
            return prefix.new_tensor(float("-inf"))
        return torch.logsumexp(torch.stack(finals), dim=0)

    def _log_partition_single(self, coarse_logits: torch.Tensor, phase_logits: torch.Tensor, length: int) -> torch.Tensor:
        cache = self._build_span_cache(coarse_logits, phase_logits, length)
        return self._log_partition_single_from_cache(*cache, length)

    def _gold_score_single_from_cache(
        self,
        prefix: torch.Tensor,
        none_unit_scores: torch.Tensor,
        segment_scores,
        states: torch.Tensor,
        length: int,
    ) -> torch.Tensor:
        dtype = prefix.dtype
        device = prefix.device
        spans = self._targets_to_spans(states, length)

        score = torch.zeros((), dtype=dtype, device=device)
        prev = START_LABEL
        for start, end, label in spans:
            tr = self._transition_score(prev, label, dtype, device)
            if tr is None:
                raise ValueError(
                    f"Gold path contains disallowed transition {prev}->{label}. "
                    f"Use --allow_same_label_segments if your labels require same-label adjacency."
                )
            score = score + tr + self._span_score_from_cache(prefix, none_unit_scores, segment_scores, start, end, label)
            prev = label
        if spans:
            score = score + self.transition_scale * self.end_transitions[prev].to(dtype=dtype, device=device)
        return score

    def _gold_score_single(self, coarse_logits: torch.Tensor, phase_logits: torch.Tensor, states: torch.Tensor, length: int) -> torch.Tensor:
        cache = self._build_span_cache(coarse_logits, phase_logits, length)
        return self._gold_score_single_from_cache(*cache, states, length)

    def neg_log_likelihood(
        self,
        coarse_logits: torch.Tensor,
        phase_logits: torch.Tensor,
        mask: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        lengths = mask.long().sum(dim=1).detach().cpu().tolist()
        losses = []
        # Float32 DP is more stable under AMP; gradients still flow through casts.
        coarse_f = coarse_logits.float()
        phase_f = phase_logits.float()
        for b, length in enumerate(lengths):
            length = int(length)
            if length <= 0:
                continue
            # Build the span cache once per sequence and reuse it for both logZ
            # and gold score.  The initial prototype built it twice, which roughly
            # doubled CPU/RAM pressure during training.
            cache = self._build_span_cache(coarse_f[b], phase_f[b], length)
            log_z = self._log_partition_single_from_cache(*cache, length)
            gold = self._gold_score_single_from_cache(*cache, targets[b], length)
            losses.append(log_z - gold)
        if not losses:
            return coarse_logits.sum() * 0.0
        return torch.stack(losses).mean()

    def _viterbi_single_from_cache(
        self,
        prefix: torch.Tensor,
        none_unit_scores: torch.Tensor,
        segment_scores,
        length: int,
    ) -> List[Tuple[int, int, int]]:
        """Return best full-covering spans for one sequence, no gradients needed."""
        dtype = prefix.dtype
        device = prefix.device
        zero = torch.zeros((), dtype=dtype, device=device)

        scores: List[List[Optional[torch.Tensor]]] = [[None for _ in range(self.num_labels)] for _ in range(length + 1)]
        back: List[List[Optional[Tuple[int, int, int, int, int]]]] = [[None for _ in range(self.num_labels)] for _ in range(length + 1)]

        for pos in range(length + 1):
            prev_candidates: List[Tuple[int, torch.Tensor]] = []
            if pos == 0:
                prev_candidates.append((START_LABEL, zero))
            for lab in range(self.num_labels):
                if scores[pos][lab] is not None:
                    prev_candidates.append((lab, scores[pos][lab]))
            if not prev_candidates:
                continue

            if pos < length:
                curr = NONE_LABEL
                span_score = none_unit_scores[pos]
                for prev_lab, prev_score in prev_candidates:
                    tr = self._transition_score(prev_lab, curr, dtype, device)
                    if tr is None:
                        continue
                    cand = prev_score + tr + span_score
                    old = scores[pos + 1][curr]
                    if old is None or cand.item() > old.item():
                        scores[pos + 1][curr] = cand
                        back[pos + 1][curr] = (pos, prev_lab, pos, pos, curr)

            max_m = min(self.max_seg_len, length - pos)
            if max_m < self.min_seg_len:
                continue
            for curr in range(1, self.num_labels):
                curr_scores_by_len = segment_scores[curr]
                if curr_scores_by_len is None:
                    continue
                for m in range(self.min_seg_len, max_m + 1):
                    scores_for_m = curr_scores_by_len[m]
                    if scores_for_m is None or pos >= scores_for_m.numel():
                        continue
                    span_score = scores_for_m[pos]
                    new_pos = pos + m
                    for prev_lab, prev_score in prev_candidates:
                        tr = self._transition_score(prev_lab, curr, dtype, device)
                        if tr is None:
                            continue
                        cand = prev_score + tr + span_score
                        old = scores[new_pos][curr]
                        if old is None or cand.item() > old.item():
                            scores[new_pos][curr] = cand
                            back[new_pos][curr] = (pos, prev_lab, pos, new_pos - 1, curr)

        best_lab: Optional[int] = None
        best_score: Optional[torch.Tensor] = None
        for lab in range(self.num_labels):
            if scores[length][lab] is None:
                continue
            cand = scores[length][lab] + self.transition_scale * self.end_transitions[lab].to(dtype=dtype, device=device)
            if best_score is None or cand.item() > best_score.item():
                best_score = cand
                best_lab = lab
        if best_lab is None:
            return [(0, length - 1, NONE_LABEL)] if length > 0 else []

        spans: List[Tuple[int, int, int]] = []
        pos = length
        lab = int(best_lab)
        while pos > 0:
            bp = back[pos][lab]
            if bp is None:
                break
            prev_pos, prev_lab, start, end, curr = bp
            spans.append((start, end, curr))
            pos = prev_pos
            lab = prev_lab if prev_lab != START_LABEL else NONE_LABEL
        spans.reverse()

        # Merge consecutive background unit spans for cleanliness.
        merged: List[Tuple[int, int, int]] = []
        for s, e, lab in spans:
            if merged and lab == NONE_LABEL and merged[-1][2] == NONE_LABEL and merged[-1][1] + 1 == s:
                merged[-1] = (merged[-1][0], e, NONE_LABEL)
            else:
                merged.append((s, e, lab))
        return merged

    def _viterbi_single(self, coarse_logits: torch.Tensor, phase_logits: torch.Tensor, length: int) -> List[Tuple[int, int, int]]:
        cache = self._build_span_cache(coarse_logits, phase_logits, length)
        return self._viterbi_single_from_cache(*cache, length)

    def _spans_to_old_state_path(self, spans: Sequence[Tuple[int, int, int]], length: int) -> List[int]:
        path = [0] * length
        for start, end, label in spans:
            if label == NONE_LABEL:
                continue
            seg_len = end - start + 1
            if label == PEPTIDE_LABEL:
                start_state = 1
                end_state = OLD_STATE_MAX_LEN
            elif label == PROPEPTIDE_LABEL:
                start_state = OLD_STATE_MAX_LEN + 1
                end_state = 2 * OLD_STATE_MAX_LEN
            else:
                continue

            if seg_len <= 0:
                continue
            if seg_len == 1:
                states = [start_state]
            elif seg_len == 2:
                states = [start_state, end_state]
            else:
                states = [start_state] + [start_state + 1] * (seg_len - 2) + [end_state]
            path[start : end + 1] = states[:seg_len]
        return path

    def viterbi_decode_batch(self, coarse_logits: torch.Tensor, phase_logits: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
        lengths = mask.long().sum(dim=1).detach().cpu().tolist()
        paths: List[List[int]] = []
        # Decoding is used only for evaluation/logging.  Keeping it under
        # no_grad prevents accidental graph construction if the caller forgets.
        with torch.no_grad():
            coarse_f = coarse_logits.detach().float()
            phase_f = phase_logits.detach().float()
            for b, length in enumerate(lengths):
                length = int(length)
                spans = self._viterbi_single(coarse_f[b], phase_f[b], length)
                paths.append(self._spans_to_old_state_path(spans, length))
        return paths

    def make_pseudo_state_probs(self, coarse_logits: torch.Tensor) -> torch.Tensor:
        """Map coarse class probabilities to old multistate probability shape.

        These are not true segmental CRF marginals. They are returned only for
        compatibility with existing logging/metric code. Segment-level metrics in
        manuscript_metrics.py use decoded paths, not these probabilities.
        """
        # The returned probabilities are used only for metrics/logging, not loss.
        # Detach to avoid keeping an unnecessary graph during validation.
        class_probs = torch.softmax(coarse_logits.detach().float(), dim=-1).to(dtype=coarse_logits.dtype)
        bsz, seq_len, _ = class_probs.shape
        out = coarse_logits.new_zeros(bsz, seq_len, self.num_states)
        out[:, :, 0] = class_probs[:, :, NONE_LABEL]
        out[:, :, 1 : OLD_STATE_MAX_LEN + 1] = class_probs[:, :, PEPTIDE_LABEL].unsqueeze(-1) / float(OLD_STATE_MAX_LEN)
        if self.num_labels == 3:
            out[:, :, OLD_STATE_MAX_LEN + 1 : 2 * OLD_STATE_MAX_LEN + 1] = (
                class_probs[:, :, PROPEPTIDE_LABEL].unsqueeze(-1) / float(OLD_STATE_MAX_LEN)
            )
        return out

    def forward(
        self,
        coarse_logits: torch.Tensor,
        phase_logits: torch.Tensor,
        mask: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        skip_marginals: bool = False,
        top_k: int = 1,
        decode: bool = True,
        return_probs: bool = True,
    ):
        if top_k != 1:
            raise NotImplementedError("RelativePositionSegmentalCRF currently supports only top_k=1 decoding")
        mask = mask.bool()

        loss = None
        if targets is not None:
            loss = self.neg_log_likelihood(coarse_logits, phase_logits, mask, targets.long())

        viterbi_paths = None
        if decode:
            viterbi_paths = self.viterbi_decode_batch(coarse_logits, phase_logits, mask)

        probs = None
        if return_probs:
            probs = self.make_pseudo_state_probs(coarse_logits)

        if targets is not None:
            return probs, viterbi_paths, loss
        return probs, viterbi_paths, None
