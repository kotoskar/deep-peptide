# Implementation plan: length-aware telescoping segmental CRF emissions

## 0. Goal

Implement a length-aware segment scoring mechanism while keeping the existing multistate CRF dynamic programming structure as much as possible.

Current DeepPeptide-style CRF uses states like:

```text
0          = NONE
1..K       = PEPTIDE length/progress states
K+1..2K    = PROPEP length/progress states
```

where usually `K = 50`.

Current model emits only coarse logits:

```text
NONE / PEPTIDE / PROPEP
```

and then copies the same `PEPTIDE` logit to all `PEPTIDE` CRF states, and the same `PROPEP` logit to all `PROPEP` states.

This plan replaces the copied peptide/propeptide emissions with **delta span emissions** so that the accumulated score along:

```text
PEP_1 -> PEP_2 -> ... -> PEP_m
```

is equal to a length-aware score of the whole candidate segment.

In other words, the ordinary CRF path should behave as if it scored a whole span `[s, e]` with known length, without needing a completely separate segmental CRF implementation.

---

## 1. Core idea

For a candidate segment:

```text
span = [s, e]
length m = e - s + 1
label y ∈ {PEPTIDE, PROPEP}
```

we define a segment/span score:

```text
SpanScore_y(s, e)
```

Then we construct CRF emissions so that this path:

```text
NONE -> y_1 -> y_2 -> ... -> y_m -> NONE/end
```

accumulates exactly:

```text
SpanScore_y(s, e)
```

for the segment part.

This is done by using telescoping delta emissions:

```text
DeltaEmission_y(i, k) = SpanScore_y(i-k+1, i) - SpanScore_y(i-k+1, i-1)
```

where:

```text
i = current sequence position
k = current length state, 1 <= k <= K
s = i - k + 1
```

For `k = 1`:

```text
DeltaEmission_y(i, 1) = SpanScore_y(i, i)
```

Then along a segment starting at `s` and ending at `e = s + m - 1`:

```text
Delta(s,1)
+ Delta(s+1,2)
+ ...
+ Delta(e,m)
= SpanScore_y(s,e)
```

because all intermediate span scores cancel out.

This means the CRF can keep using its usual transition structure, but the emission values now encode a full length-aware segment score.

---

## 2. Position function requirement

The model should predict, for each residue and each segment label, a preferred relative coordinate:

```text
r_pred_y[i] ∈ [0, 1]
```

Interpretation:

```text
0.0 = this residue looks like segment beginning
0.5 = this residue looks like segment middle
1.0 = this residue looks like segment end
```

For a candidate span `[s, e]`, the actual relative coordinate of residue `i` inside the span is:

```text
t_rel(i; s,e) = (i - s) / (e - s)
```

For a length-1 segment, use:

```text
t_rel = 0.5
```

The initial position score function should be:

```text
position_score(r_pred, t_rel) = -abs(t_rel - r_pred) / tau
```

where `tau` is a configurable positive scalar.

Important: use this as a **log-score / potential**, not as a probability.

Do not use:

```text
1 - abs(t_rel - r_pred)
```

because that gives positive bonuses to almost every position and can accidentally favor longer spans.

---

## 3. Make the position function replaceable

Do not hide the position scoring function inline inside the CRF.

Create a separate module/function, for example:

```python
# src/models/position_scores.py

def neg_abs_position_score(r_pred, t_rel, tau=0.25):
    return -torch.abs(t_rel - r_pred) / tau
```

Recommended API:

```python
def compute_position_score(
    r_pred: torch.Tensor,
    t_rel: torch.Tensor,
    mode: str = "neg_abs",
    tau: float = 0.25,
    **kwargs,
) -> torch.Tensor:
    ...
```

Supported initial modes:

```text
neg_abs:
    score = -abs(t_rel - r_pred) / tau

neg_squared:
    optional future mode
    score = -((t_rel - r_pred) ** 2) / (2 * tau ** 2)

none:
    score = 0
```

The code should expose CLI/config options:

```bash
--position_score_mode neg_abs
--position_score_tau 0.25
--position_score_scale 0.25
```

`position_score_scale` multiplies the positional contribution before adding it into the span score.

---

## 4. Model outputs

The neural model should output the usual coarse emissions:

```text
coarse_logits: [B, L, 3]
```

with channels:

```text
0 = NONE
1 = PEPTIDE
2 = PROPEP
```

Add a relative-position head:

```text
position_raw: [B, L, 2]
```

where:

```text
position_raw[..., 0] = raw preferred relative coordinate for PEPTIDE
position_raw[..., 1] = raw preferred relative coordinate for PROPEP
```

Convert to `[0,1]` with sigmoid:

```python
r_pred = torch.sigmoid(position_raw)
```

Shape:

```text
r_pred: [B, L, 2]
```

Optional but recommended: initialize the final layer of the position head near zero so initial `r_pred ≈ 0.5`.

This gives a neutral middle-position prior at the beginning of training.

---

## 5. Span score definition

For label `y ∈ {PEPTIDE, PROPEP}`:

```text
SpanScore_y(s,e) =
    sum_{i=s..e} coarse_logit_y[i]
  + position_score_scale * sum_{i=s..e} position_score(r_pred_y[i], t_rel(i; s,e))
  + optional_length_bias_y[length]
  + optional_start_bias_y[s]
  + optional_end_bias_y[e]
```

For the first implementation, use only:

```text
sum coarse logits
+ position_score_scale * sum position_score
```

Do not add start/end/length terms until the basic version is verified.

### Initial span score

```text
SpanScore_y(s,e) =
    coarse_sum_y(s,e)
  + beta * relpos_sum_y(s,e)
```

where:

```text
beta = position_score_scale
```

---

## 6. Precompute span scores

For each batch item, label, end position, and length:

```text
span_scores_y[b, end, k]
```

where:

```text
k = 1..K
start = end - k + 1
```

Invalid spans where `start < 0` must be set to `-inf` or masked before use.

However, for delta computation, it may be easier to keep invalid spans as `0` internally and mask the final delta emissions to `-inf` for invalid states.

Recommended tensor shape:

```text
span_scores: [B, L, 2, K]
```

where label index:

```text
0 = PEPTIDE
1 = PROPEP
```

### Relative coordinates

For each length `k`, build a vector:

```text
t_rel[k, j]
```

where:

```text
j = 0..k-1
```

and:

```text
if k == 1:
    t_rel = 0.5
else:
    t_rel[j] = j / (k - 1)
```

For each candidate span ending at `end`, the residues are:

```text
start = end-k+1
positions start..end
```

Evaluate:

```text
position_score(r_pred_y[position], t_rel[j])
```

and sum over `j`.

### Simple implementation first

A loop over `k = 1..K` is acceptable for the first version because `K=50`.

Pseudo-code:

```python
span_scores = torch.full((B, L, 2, K), -inf, device=device)

for k in range(1, K + 1):
    # windows over sequence dimension, length k
    # start = end-k+1
    # valid ends are k-1..L-1

    # coarse sums can be obtained with cumsum or unfold
    # relpos sums can be obtained with unfold over r_pred and t_rel vector

    span_scores[:, k-1:, label_idx, k-1] = computed_scores
```

For prototype clarity, using `unfold` is fine:

```python
# r_label: [B, L]
# coarse_label: [B, L]
windows_r = r_label.unfold(dimension=1, size=k, step=1)       # [B, L-k+1, k]
windows_c = coarse_label.unfold(dimension=1, size=k, step=1)  # [B, L-k+1, k]

t_rel = make_t_rel(k).view(1, 1, k)                          # [1,1,k]
pos = compute_position_score(windows_r, t_rel, mode, tau)     # [B,L-k+1,k]

score = windows_c.sum(-1) + beta * pos.sum(-1)                # [B,L-k+1]

# These spans end at positions k-1..L-1
span_scores[:, k-1:, label_idx, k-1] = score
```

This is differentiable. Gradients will flow through `windows_r`, `position_score`, sums, deltas, CRF forward, and CRF loss.

---

## 7. Convert span scores to delta emissions

Build CRF state emissions:

```text
state_emissions: [B, L, 1 + 2K]
```

State mapping:

```text
NONE state        = 0
PEPTIDE state k   = k,       k=1..K
PROPEP state k    = K + k,   k=1..K
```

NONE emission remains the ordinary coarse NONE logit:

```python
state_emissions[:, :, 0] = coarse_logits[:, :, NONE]
```

For PEPTIDE/PROPEP states, use delta emissions.

For a label `y`, end position `i`, length `k`:

```text
Delta_y(i,k) = SpanScore_y(i-k+1, i) - SpanScore_y(i-k+1, i-1)
```

For `k=1`:

```text
Delta_y(i,1) = SpanScore_y(i,i)
```

Implementation detail:

`SpanScore_y(i-k+1, i-1)` is the span with:

```text
same start = i-k+1
previous end = i-1
previous length = k-1
```

So it is available at:

```text
span_scores[:, i-1, y, k-2]
```

for `k > 1`.

Pseudo-code:

```python
delta = torch.full_like(span_scores, -inf)  # [B,L,2,K]

# k = 1
delta[:, :, :, 0] = span_scores[:, :, :, 0]

# k > 1
for k in range(2, K + 1):
    curr = span_scores[:, :, :, k-1]        # [B,L,2]
    prev = torch.empty_like(curr)
    prev[:, 0, :] = 0 or invalid            # no valid previous span at i=0
    prev[:, 1:, :] = span_scores[:, :-1, :, k-2]
    delta[:, :, :, k-1] = curr - prev
```

But invalid entries must be masked to `-inf` after subtraction.

Valid condition:

```text
i - k + 1 >= 0
```

So for each `k`, positions `i < k-1` are invalid.

Assign to CRF states:

```python
state_emissions[:, :, 1:K+1] = delta[:, :, PEPTIDE_IDX, :]
state_emissions[:, :, K+1:2*K+1] = delta[:, :, PROPEP_IDX, :]
```

---

## 8. Why telescoping works

For a PEPTIDE span `[s,e]` with length `m`:

```text
position s     uses PEP_1 emission = SpanScore(s,s)
position s+1   uses PEP_2 emission = SpanScore(s,s+1) - SpanScore(s,s)
position s+2   uses PEP_3 emission = SpanScore(s,s+2) - SpanScore(s,s+1)
...
position e     uses PEP_m emission = SpanScore(s,e) - SpanScore(s,e-1)
```

Summing all emissions:

```text
SpanScore(s,s)
+ SpanScore(s,s+1) - SpanScore(s,s)
+ SpanScore(s,s+2) - SpanScore(s,s+1)
+ ...
+ SpanScore(s,e) - SpanScore(s,e-1)
= SpanScore(s,e)
```

So the standard CRF path accumulates the same score as a direct segmental span model.

---

## 9. Transition constraints

This method requires strict monotonic segment states.

Allowed transitions should include:

```text
NONE -> NONE
NONE -> PEP_1
NONE -> PRO_1

PEP_k -> PEP_{k+1}, for k=1..K-1
PRO_k -> PRO_{k+1}, for k=1..K-1

PEP_k -> NONE, for k=1..K
PRO_k -> NONE, for k=1..K

Optional if current baseline allows adjacent labeled segments:
PEP_k -> PRO_1
PRO_k -> PEP_1
PEP_k -> PEP_1      # adjacent peptide segments, only if baseline supports this
PRO_k -> PRO_1      # adjacent propeptide segments, only if baseline supports this
```

Disallow transitions such as:

```text
NONE -> PEP_17
PEP_3 -> PEP_8
PEP_8 -> PEP_8
PRO_4 -> PRO_2
```

because they break the interpretation:

```text
state y_k = current segment has length exactly k
```

If the existing CRF already enforces this, keep it.

---

## 10. End-of-sequence handling

No special `segment bonus on exit` is needed.

The full segment score is already accumulated by the time the path reaches `PEP_m` or `PRO_m`.

Therefore a protein may end in:

```text
NONE
PEP_m
PRO_m
```

and the score is complete.

Do not require a final transition to `NONE` to add the segment score.

This avoids the failure mode of exit-bonus hybrids where a terminal PEPTIDE/PROPEP segment would not receive its segment-level score.

---

## 11. Loss and numerical stability

Keep the ordinary CRF negative log-likelihood:

```text
loss = logZ - gold_path_score
```

where:

```text
logZ = logsumexp over all valid state paths
```

Everything must be computed in log-space / score-space.

Do not multiply probabilities.

Use:

```python
torch.logsumexp(...)
```

for the forward algorithm.

Use `-inf` or a very negative number for invalid emissions/transitions.

Recommended:

```python
NEG_INF = -1e4 or -1e9 depending on existing CRF conventions
```

If the existing CRF uses a specific negative sentinel, reuse it.

---

## 12. Gold labels

The existing multistate gold labels should still work if they already encode:

```text
PEP_1, PEP_2, ..., PEP_m
PRO_1, PRO_2, ..., PRO_m
```

for labeled segments.

The gold path score under the new delta emissions will automatically telescope into the correct span score.

No need to rewrite labels initially.

Need to verify:

1. Gold PEPTIDE segment of length `m` uses states `PEP_1..PEP_m`.
2. Gold PROPEP segment of length `m` uses states `PRO_1..PRO_m`.
3. Lengths above `K` are handled exactly as current code handles them.

If current code has special behavior for segments longer than 50, preserve it for the first implementation.

---

## 13. Handling long segments

If `K=50`, the model can directly represent labeled spans up to length 50.

Need to check current baseline behavior for longer PEPTIDE/PROPEP spans.

Possible policies:

1. Preserve existing truncation/clipping behavior.
2. Increase `K` if memory allows.
3. Fall back to ordinary coarse emissions for states beyond `K` if such states exist.
4. Exclude or specially handle overlength segments exactly as current training does.

Do not silently change the dataset semantics.

---

## 14. Optional auxiliary relative-position loss

Not required for the first implementation, but useful.

For each residue inside a gold PEPTIDE/PROPEP segment:

```text
r_true = (i - s) / (e - s)
```

For length 1:

```text
r_true = 0.5
```

Then:

```text
L_pos = SmoothL1(r_pred_y[i], r_true)
```

Only apply to residues inside the corresponding label.

Total loss:

```text
L = L_CRF + lambda_pos * L_pos
```

Expose:

```bash
--relative_position_loss_lambda 0.0
```

Default should be `0.0` for the first pure-CRF experiment.

Suggested later values:

```text
0.005, 0.01, 0.02
```

---

## 15. Initial CLI/config parameters

Add options similar to:

```bash
--model lstmcnncrf_telescoping_segmental
--position_score_mode neg_abs
--position_score_tau 0.25
--position_score_scale 0.25
--relative_position_loss_lambda 0.0
--segmental_max_len 50
```

Recommended initial run:

```bash
--position_score_mode neg_abs
--position_score_tau 0.25
--position_score_scale 0.25
--relative_position_loss_lambda 0.0
--segmental_max_len 50
```

If unstable or too conservative:

```bash
--position_score_scale 0.1
```

If positional signal is too weak:

```bash
--position_score_scale 0.5
```

---

## 16. Recommended implementation structure

Suggested files:

```text
src/models/position_scores.py
src/models/crf_models.py
src/models/__init__.py
src/train_loop_crf.py
```

Add a model class such as:

```python
class LSTMCNNCRFTelescopingSegmental(nn.Module):
    ...
```

or extend the existing LSTMCNNCRF class behind a flag if cleaner.

The model forward should return:

```python
{
    "state_emissions": state_emissions,        # [B,L,1+2K]
    "coarse_logits": coarse_logits,            # [B,L,3]
    "r_pred": r_pred,                          # [B,L,2]
    "span_scores": optional_for_debug,         # [B,L,2,K]
    "delta_emissions": optional_for_debug,     # [B,L,2,K]
}
```

Only return debug tensors when requested if memory becomes an issue.

---

## 17. Unit tests

### Test 1: telescoping identity

Construct random span scores manually.

For a fixed start `s`, end `e`, label `y`, compute deltas and verify:

```text
sum_{k=1..m} Delta(s+k-1, k) == SpanScore(s,e)
```

within numerical tolerance.

### Test 2: length-1 segment

For `k=1`, verify:

```text
Delta(i,1) == SpanScore(i,i)
```

### Test 3: invalid states

For every `k`, positions:

```text
i < k - 1
```

must be invalid for state `y_k`.

### Test 4: terminal segment

Create a toy sequence ending in PEPTIDE:

```text
NONE, PEP_1, PEP_2, PEP_3
```

Verify the path score includes:

```text
SpanScore(last 3 residues, PEPTIDE)
```

without requiring transition to NONE.

### Test 5: gradient flow

Run a tiny forward/backward pass and verify gradients exist for:

```text
position head parameters
coarse emission head parameters
feature extractor parameters
```

### Test 6: equivalence when scale = 0

When:

```text
position_score_scale = 0
```

and span score uses only coarse logits, verify behavior is equivalent to summing coarse logits over segment via deltas.

This may not be exactly identical to old repeated-emission CRF if old CRF had additional learned internal transition biases, but the emission contribution should match.

---

## 18. Important notes

1. Do not detach span scores or delta emissions.
2. Do not use argmax inside training forward.
3. Do not convert scores to probabilities before CRF.
4. Keep all CRF computation in log-space.
5. Keep the position scoring function explicit and configurable.
6. Start with small `position_score_scale`.
7. Preserve existing transition constraints unless they violate the strict length-state interpretation.

---

## 19. Short conceptual summary

Old CRF emission:

```text
Emission(i, PEP_k) = PEPTIDE_logit(i)
```

New telescoping emission:

```text
Emission(i, PEP_k) = SpanScore_PEP(i-k+1, i) - SpanScore_PEP(i-k+1, i-1)
```

Therefore:

```text
PEP_1 -> PEP_2 -> ... -> PEP_m
```

accumulates:

```text
SpanScore_PEP(s,e)
```

where every residue is evaluated at its relative coordinate inside the candidate segment.

Initial relative-position score:

```text
position_score(r_pred, t_rel) = -abs(t_rel - r_pred) / tau
```

This solves the fixed-length-state problem without assigning absolute biological meaning to `PEP_37`.
