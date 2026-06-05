# Length-aware Segmental CRF with Relative-Position Segment Scoring

## 0. Purpose

This document specifies a proposed replacement / alternative for the current DeepPeptide-style multistate CRF decoding layer.

The goal is to avoid the current weakness where a single coarse `PEPTIDE` or `PROPEP` emission is copied across many latent CRF states (`pep_1 ... pep_50`, `propep_1 ... propep_50`). Instead, the model should score candidate peptide/propeptide segments as whole spans with known start, end, length, and relative residue positions.

The proposed model is a **length-aware segmental CRF**.

Core idea:

```text
Instead of building a PEPTIDE segment residue-by-residue through states
Pep1 -> Pep2 -> ... -> PepK,

score each candidate segment [s, e] directly.

When scoring [s, e], the segment length is known, so each residue t in [s, e]
has a known relative coordinate:

    r(t; s, e) = (t - s) / (e - s)

or a special value for length-1 segments.
```

This lets the model ask:

```text
If residues s..e form one PEPTIDE segment, how compatible is each residue
with its relative role inside that segment?
```

This is more principled than assigning absolute meanings to `pep_1 ... pep_50` states.

---

## 1. Current architecture near the CRF

Current DeepPeptide-style architecture:

```text
ESM2 embeddings [B, L, C]
-> LSTMCNN / feature extractor
-> coarse emission head
-> logits [B, L, 3]

classes:
    0 = None / background
    1 = PEPTIDE
    2 = PROPEP

Then:
    logits [B, L, 3]
    -> repeated into CRF states [B, L, 101]

states:
    0       = None
    1..50   = PEPTIDE latent states
    51..100 = PROPEP latent states
```

Important weakness:

```text
For every residue position i:

    emission[i, pep_1]  = peptide_logit[i]
    emission[i, pep_2]  = peptide_logit[i]
    ...
    emission[i, pep_50] = peptide_logit[i]

The neural model says only:

    "this residue is peptide-like"

It does not directly say:

    "this residue looks like the start / early / middle / late / end
     of a peptide segment."
```

The CRF has many latent states, but the neural emissions do not meaningfully parameterize those states.

---

## 2. Why naive 50-state positional prediction is not enough

A tempting fix is:

```text
h_i -> 50 PEPTIDE position logits
h_i -> 50 PROPEP position logits
```

But this is problematic if interpreted as absolute positions:

```text
Pep37 = 37th amino acid of the peptide
```

Peptides/propeptides have variable lengths. For a short peptide of length 8, there is no absolute 37th residue. For a long peptide of length 45, state 37 is a late-internal residue. Therefore the useful concept is not absolute position, but **relative position inside a candidate segment**.

Correct framing:

```text
For a candidate segment [s, e], residue t has relative coordinate:

    r = (t - s) / (e - s)

Then the model evaluates whether residue t is compatible with relative
position r in that segment.
```

This requires knowing the candidate segment length. Therefore the clean solution is to score full segments directly.

---

## 3. Proposed model: segmental CRF

Instead of decoding through residue-level latent states, decode through a sequence of labeled spans.

Allowed labels:

```text
N = None / background
P = PEPTIDE segment
R = PROPEP segment
```

A full path is a segmentation of the sequence into consecutive spans, for example:

```text
[1..20]   None
[21..34]  PEPTIDE
[35..70]  None
[71..110] PROPEP
[111..118] PEPTIDE
[119..L]  None
```

The score of a path is the sum of span scores plus optional transition scores between span labels.

```text
PathScore(path) =
    sum over spans SpanScore(start, end, label)
  + sum over adjacent spans TransitionScore(prev_label, next_label)
```

The loss is the usual CRF negative log-likelihood, but over segmentations rather than residue-level state paths:

```text
Loss = logsumexp_{all valid segmentations} PathScore(segmentation)
     - PathScore(gold_segmentation)
```

This is a **semi-Markov CRF / segmental CRF**.

---

## 4. Indexing conventions

Use explicit indexing to avoid bugs.

Recommended implementation indexing:

```text
Python tensors use 0-based residue indices.
Sequence length = L.
Residue positions are 0, 1, ..., L-1.

A segment [s, e] is inclusive:
    0 <= s <= e < L

Segment length:
    m = e - s + 1
```

For a segment of length `m > 1`, relative coordinate of residue `t` inside `[s, e]`:

```text
r(t; s, e) = (t - s) / (m - 1)
```

So:

```text
t = s -> r = 0.0
t = e -> r = 1.0
```

For `m == 1`, use:

```text
r = 0.5
```

or define a special single-residue handling. For the first prototype, `r = 0.5` is acceptable.

---

## 5. Network outputs

The model should produce residue-level hidden representations as before:

```text
x: [B, L, input_dim]
h: [B, L, hidden_dim]
```

From `h`, produce at least the following outputs.

### 5.1 Coarse residue logits

```text
coarse_logits: [B, L, 3]
```

Classes:

```text
0 = None
1 = PEPTIDE
2 = PROPEP
```

These can reuse the existing emission head.

### 5.2 Relative-position compatibility function

We need a function:

```text
F_label(b, t, r)
```

meaning:

```text
For batch item b and residue t, how compatible is this residue with being
at relative coordinate r inside a segment of type label?
```

Labels requiring relative-position compatibility:

```text
PEPTIDE
PROPEP
```

`None` does not need a relative-position function.

There are several possible parameterizations. For the first implementation, use a simple basis-function head.

---

## 6. Recommended first implementation: phase-basis relative scoring

Use `K` smooth phase bases over relative coordinate `r in [0, 1]`.

Recommended first value:

```text
K = 5
phase centers = [0.0, 0.25, 0.5, 0.75, 1.0]
```

Interpretation:

```text
0.0  = start
0.25 = early
0.5  = middle
0.75 = late
1.0  = end
```

### 6.1 Model head

For each residue, predict phase logits for PEPTIDE and PROPEP:

```text
phase_logits_pep:    [B, L, K]
phase_logits_propep: [B, L, K]
```

or combined:

```text
phase_logits: [B, L, 2, K]
```

where label index 0 = PEPTIDE, 1 = PROPEP.

### 6.2 Basis functions

For a relative coordinate `r`, define fixed Gaussian/RBF basis values:

```text
basis_k(r) = exp(- (r - center_k)^2 / (2 * sigma_basis^2))
```

Recommended first value:

```text
sigma_basis = 0.20
```

Optionally normalize:

```text
basis_k(r) = basis_k(r) / sum_j basis_j(r)
```

Recommended for stability: normalize basis values so they sum to 1.

### 6.3 Relative-position score

For a residue `t` and relative coordinate `r`:

```text
RelScore_PEPTIDE[t, r] = sum_k phase_logits_pep[t, k] * basis_k(r)
RelScore_PROPEP[t, r] = sum_k phase_logits_propep[t, k] * basis_k(r)
```

This means the model does not classify residues into absolute states. It predicts phase evidence, and the segmental CRF queries that evidence at the relative coordinate implied by a candidate segment.

---

## 7. Span score

For a candidate segment `[s, e]` with label `label`, compute:

```text
SpanScore(s, e, label)
```

### 7.1 None / background span score

For a None span:

```text
SpanScore(s, e, None) = sum_{t=s..e} coarse_logits[t, None]
```

No relative-position score is needed.

### 7.2 PEPTIDE span score

For a PEPTIDE span:

```text
m = e - s + 1

SpanScore(s, e, PEPTIDE) =
    sum_{t=s..e} coarse_logits[t, PEPTIDE]
  + beta_rel * sum_{t=s..e} RelScore_PEPTIDE[t, r(t; s, e)]
  + beta_start * StartScore_PEPTIDE[s]
  + beta_end   * EndScore_PEPTIDE[e]
  + LengthPrior_PEPTIDE[m]
```

For the first prototype, `StartScore` and `EndScore` can be omitted if not already available:

```text
beta_start = 0
beta_end = 0
```

Minimum viable span score:

```text
SpanScore(s, e, PEPTIDE) =
    sum_{t=s..e} coarse_logits[t, PEPTIDE]
  + beta_rel * sum_{t=s..e} RelScore_PEPTIDE[t, r(t; s, e)]
  + LengthPrior_PEPTIDE[m]
```

### 7.3 PROPEP span score

Analogously:

```text
SpanScore(s, e, PROPEP) =
    sum_{t=s..e} coarse_logits[t, PROPEP]
  + beta_rel * sum_{t=s..e} RelScore_PROPEP[t, r(t; s, e)]
  + LengthPrior_PROPEP[m]
```

### 7.4 Length prior

Use learned length priors for segment labels:

```text
LengthPrior_PEPTIDE: [max_seg_len + 1]
LengthPrior_PROPEP:  [max_seg_len + 1]
```

Index by length `m`.

For `None`, either:

1. Do not use a length prior, or
2. Allow arbitrary-length None runs through per-residue recurrence instead of enumerating None spans.

Recommended first implementation: treat background at residue level, not as enumerated long spans. See DP below.

---

## 8. Maximum segment length

The current multistate CRF uses 50 PEPTIDE states and 50 PROPEP states. Therefore the natural first setting is:

```text
max_seg_len = 50
```

Candidate PEPTIDE/PROPEP spans longer than 50 are not allowed in the first prototype.

If the dataset contains gold PEPTIDE/PROPEP segments longer than 50, there are three options:

1. Keep the existing behavior and treat them as unsupported / clipped / invalid according to current preprocessing.
2. Increase `max_seg_len`.
3. Add overflow handling later.

For the first implementation, match the current model assumption:

```text
max_seg_len = 50
```

and explicitly assert/log if gold segments exceed this.

---

## 9. Dynamic programming: forward normalizer

We need compute:

```text
logZ = logsumexp over all valid segmentations
```

A simple and safe DP uses end-position recurrence.

Define:

```text
alpha[i, y]
```

where:

```text
alpha[i, y] = logsumexp score of all segmentations covering residues 0..i,
              whose last emitted label is y.
```

Labels:

```text
0 = None
1 = PEPTIDE
2 = PROPEP
```

But it is often cleaner to use prefix length indexing:

```text
dp[pos, y]
```

where:

```text
dp[pos, y] = logsumexp score of all segmentations covering residues 0..pos-1,
             with last label y.

pos ranges from 0..L.
pos = 0 means empty prefix.
```

Use prefix indexing to avoid off-by-one errors.

### 9.1 Initialization

Introduce a virtual START label.

```text
dp[0, START] = 0
dp[0, None] = -inf
dp[0, PEPTIDE] = -inf
dp[0, PROPEP] = -inf
```

Alternatively keep a separate vector over real labels and handle transitions from START explicitly.

### 9.2 Allowed transitions

At the segment level, each new span has a label `curr_label`.

Allowed transitions should reflect the existing model constraints.

For the first prototype, allow:

```text
START -> None
START -> PEPTIDE
START -> PROPEP

None -> None
None -> PEPTIDE
None -> PROPEP

PEPTIDE -> None
PEPTIDE -> PROPEP
PEPTIDE -> PEPTIDE     optional, see below

PROPEP -> None
PROPEP -> PEPTIDE
PROPEP -> PROPEP       optional, see below
```

Important design choice:

If adjacent same-label spans are allowed, the model could split one long PEPTIDE span into multiple PEPTIDE spans. That is usually undesirable because segmentation boundaries would become ambiguous.

Recommended first implementation:

```text
Disallow PEPTIDE -> PEPTIDE
Disallow PROPEP -> PROPEP
```

If the dataset has adjacent same-type gold segments with no background between them, handle them explicitly in preprocessing or allow same-label transitions only if such cases are required. But default should be to disallow same-label adjacent spans to avoid degenerate splitting.

Recommended allowed transitions:

```text
START -> None
START -> PEPTIDE
START -> PROPEP

None -> None
None -> PEPTIDE
None -> PROPEP

PEPTIDE -> None
PEPTIDE -> PROPEP

PROPEP -> None
PROPEP -> PEPTIDE
```

Transition scores:

```text
trans[prev_label, curr_label]
```

Use learned transition scores for real labels and learned/start scores for START if desired.

For the first version, transitions can be initialized to 0 for allowed transitions and `-inf` for disallowed transitions.

### 9.3 Background handling

There are two possible approaches.

#### Option A: enumerate None spans

Enumerate None spans like PEPTIDE/PROPEP spans, possibly with no max length.

This is inefficient if None spans are long.

#### Option B: residue-level None continuation

Recommended first implementation:

Background can continue one residue at a time:

```text
None at pos -> None at pos+1
```

But PEPTIDE/PROPEP are segment jumps.

This gives a hybrid DP:

1. Add one background residue.
2. Add a PEPTIDE span of length 1..max_seg_len.
3. Add a PROPEP span of length 1..max_seg_len.

This avoids enumerating arbitrary long None spans.

### 9.4 Recommended DP recurrence

Use prefix length `pos`, where next uncovered residue is `pos`.

Maintain:

```text
dp[pos, last_label]
```

meaning residues `0..pos-1` are covered.

At each `pos`, extend in three ways.

#### 9.4.1 Add one None residue

New span label is `None`, length 1, covering `[pos, pos]`.

```text
new_pos = pos + 1
span_score = coarse_logits[pos, None]

for prev_label in labels_at_dp[pos]:
    if transition prev_label -> None is allowed:
        dp[new_pos, None] = logaddexp(
            dp[new_pos, None],
            dp[pos, prev_label] + trans[prev_label, None] + span_score
        )
```

Important: this treats every background residue as a separate None span. Therefore `None -> None` transition would be applied at every background residue.

To avoid penalizing/rewarding long background runs through repeated transition scores, either:

1. Set `trans[None, None] = 0` and keep it fixed, or
2. Handle None continuation separately without adding transition cost when `prev_label == None`.

Recommended first implementation:

```text
If prev_label == None and curr_label == None:
    transition cost = 0
else:
    transition cost = learned transition score
```

#### 9.4.2 Add PEPTIDE segment

For length `m = 1..max_seg_len`:

```text
s = pos
e = pos + m - 1
new_pos = pos + m
```

Only valid if:

```text
new_pos <= L
```

Compute:

```text
span_score = SpanScore(s, e, PEPTIDE)
```

Update:

```text
for prev_label:
    if transition prev_label -> PEPTIDE is allowed:
        dp[new_pos, PEPTIDE] = logaddexp(
            dp[new_pos, PEPTIDE],
            dp[pos, prev_label] + trans[prev_label, PEPTIDE] + span_score
        )
```

#### 9.4.3 Add PROPEP segment

Analogous:

```text
span_score = SpanScore(s, e, PROPEP)
```

Update `dp[new_pos, PROPEP]`.

### 9.5 Termination

At the end:

```text
logZ = logsumexp over dp[L, last_label] + optional end_transition[last_label]
```

Do not require the sequence to end in `None`.

This explicitly fixes the issue with exit-bonus approaches: if the protein ends inside a PEPTIDE/PROPEP segment, the segment score is still added when the segment span is created. There is no missing bonus at sequence end.

---

## 10. Gold path score

Gold labels are currently residue-level/multistate labels or segment annotations.

For segmental CRF, convert gold labels into a list of spans:

```text
[(s1, e1, label1), (s2, e2, label2), ...]
```

The spans must cover the full sequence without gaps:

```text
first span starts at 0
last span ends at L-1
next_s = prev_e + 1
```

Background spans should be included explicitly as `None` spans, but their score is computed as sum of per-residue None logits.

Gold score:

```text
gold_score = 0
prev_label = START

for span in gold_spans:
    curr_label = span.label
    gold_score += transition_score(prev_label, curr_label)
    gold_score += SpanScore(span.s, span.e, curr_label)
    prev_label = curr_label

gold_score += optional_end_transition(prev_label)
```

For `None` spans, if using residue-level None continuation in the normalizer, make the gold score consistent:

Recommended consistency rule:

```text
For a None span [s, e]:
    add transition into None once when entering from non-None/START;
    add coarse_logits[t, None] for every t in [s, e];
    do not add repeated None->None transition costs inside the run.
```

Therefore the DP should also not add learned `None->None` transition repeatedly. Keep `None->None` continuation cost fixed at 0.

---

## 11. Viterbi decoding

Use the same DP as the forward normalizer, replacing `logaddexp/logsumexp` with `max`.

Store backpointers:

```text
backpointer[new_pos, curr_label] = (pos, prev_label, segment_start, segment_end, curr_label)
```

For a None residue extension:

```text
segment_start = pos
segment_end = pos
curr_label = None
```

After decoding, consecutive None single-residue spans should be merged into one None span for output cleanliness.

For PEPTIDE/PROPEP jumps, store the full segment:

```text
segment_start = pos
segment_end = new_pos - 1
curr_label = PEPTIDE or PROPEP
```

Termination:

```text
best_last_label = argmax_l dp_viterbi[L, l] + optional_end_transition[l]
```

Backtrace until `pos = 0`.

Output predicted PEPTIDE/PROPEP spans. Ignore / merge None spans.

---

## 12. Efficient span-score computation

Naive computation of every span score loops over:

```text
B * L * max_seg_len * segment_length
```

With `max_seg_len = 50`, this may still be acceptable for a first prototype, but can be optimized later.

### 12.1 Coarse logit sums

Use prefix sums.

For each class `c`:

```text
prefix_c[0] = 0
prefix_c[t+1] = prefix_c[t] + coarse_logits[t, c]

sum_{u=s..e} coarse_logits[u, c] = prefix_c[e+1] - prefix_c[s]
```

### 12.2 Relative-position scores

For each candidate length `m`, relative coordinates are fixed:

```text
r_j = j / (m - 1), j = 0..m-1
```

For `m = 1`, use `r_0 = 0.5`.

Precompute basis values:

```text
basis_by_length[m, j, k]
```

where:

```text
m = 1..max_seg_len
j = 0..m-1
k = 0..K-1
```

Then for a span `[s, e]` of length `m`:

```text
rel_score = sum_{j=0..m-1} sum_k phase_logits[s+j, label, k] * basis_by_length[m, j, k]
```

This can be implemented with loops first, then optimized with convolution/unfold/einsum later.

First prototype priority: correctness over speed.

---

## 13. Masking and padding

Sequences in a batch have different lengths.

Inputs:

```text
mask: [B, Lmax]
lengths: [B]
```

For each batch item `b`, run DP only up to `L = lengths[b]`.

Do not allow spans crossing beyond `L`.

Padded residues must not contribute to span scores, forward normalizer, gold score, or Viterbi.

Simplest first implementation:

```text
Loop over batch items independently.
```

This is acceptable for correctness. Later, vectorize over batch.

---

## 14. Loss

For each sequence:

```text
nll_b = logZ_b - gold_score_b
```

Batch loss:

```text
loss = mean_b nll_b
```

Optional auxiliary losses can be added later:

```text
loss_total = segmental_crf_nll + lambda_phase * phase_aux_loss
```

But the first implementation should work with only the segmental CRF NLL, because relative-position phase scores already participate directly in the segment scores and therefore in decoding.

---

## 15. Optional auxiliary phase loss

This is optional and should not be required for the first working prototype.

If used, construct soft phase targets for residues inside gold PEPTIDE/PROPEP segments.

For a gold segment `[s, e]` of label `label`:

```text
m = e - s + 1
r_t = (t - s) / (m - 1), if m > 1
r_t = 0.5, if m == 1
```

Define soft target over phase centers:

```text
q_k ∝ exp(- (r_t - center_k)^2 / (2 * sigma_target^2))
```

Then:

```text
phase_aux_loss = CE(q, softmax(phase_logits[t, label, :]))
```

Do not apply this loss to background residues.

Recommended initial setting if used:

```text
lambda_phase = 0.01 or 0.02
sigma_target = 0.20
```

But again: start without auxiliary phase loss if possible.

---

## 16. Constraints and biological label rules

Need to match existing evaluation semantics.

Recommended first constraints:

```text
PEPTIDE and PROPEP segments cannot exceed max_seg_len.
Background can have arbitrary length.
Segments cover the full sequence.
No overlaps.
No gaps.
Adjacent same-label PEPTIDE->PEPTIDE and PROPEP->PROPEP are disallowed by default.
Sequence may start with any label.
Sequence may end with any label.
```

If current gold labels allow PEPTIDE directly adjacent to PEPTIDE as two separate annotations, then the no-same-label rule would merge them. This should be checked.

If needed, add an explicit boundary marker or allow same-label adjacency only for gold scoring and decoding, but beware of degeneracy.

---

## 17. Relationship to current CRF

Current CRF path for a peptide of length `m` ending at `e`:

```text
None -> Pep1 -> Pep2 -> ... -> Pepm -> None
```

Current score approximately:

```text
sum_{t=s..e} coarse_logits[t, PEPTIDE]
+ transition terms
```

New segmental CRF score:

```text
SegmentScore(s, e, PEPTIDE) =
    sum_{t=s..e} coarse_logits[t, PEPTIDE]
  + relative-position compatibility evaluated with known length m
  + length prior
  + optional boundary terms
```

The new model keeps the useful idea that PEPTIDE/PROPEP are contiguous spans, but avoids pretending that local state `Pep_k` can be meaningfully scored before the final segment length is known.

---

## 18. Minimal implementation plan

### Step 1: Add model class

Add a new model, for example:

```text
lstmcnn_segmental_crf
```

or:

```text
lstmcnncrf_segmental_relative
```

It should reuse:

```text
embedding input
feature extractor
coarse emission head
```

Add:

```text
phase head [B, L, 2, K]
length priors for PEPTIDE and PROPEP
segmental CRF loss/decoder module
```

### Step 2: Implement span scoring

Function signature example:

```python
def compute_span_score(
    coarse_logits,      # [L, 3]
    phase_logits,       # [L, 2, K]
    s: int,
    e: int,
    label: int,
) -> torch.Tensor:
    ...
```

Labels:

```text
0 = None
1 = PEPTIDE
2 = PROPEP
```

For `None`, return sum of None logits over `[s, e]`.

For PEPTIDE/PROPEP, return coarse sum + relative score + length prior.

### Step 3: Implement forward DP

Function:

```python
def segmental_crf_log_partition(
    coarse_logits,      # [L, 3]
    phase_logits,       # [L, 2, K]
    length: int,
) -> torch.Tensor:
    ...
```

Use prefix-index DP:

```text
dp[pos, label]
```

Include START handling.

### Step 4: Implement gold score

Function:

```python
def segmental_crf_gold_score(
    coarse_logits,
    phase_logits,
    gold_spans,
    length,
) -> torch.Tensor:
    ...
```

Need reliable conversion from existing labels to full-covering spans.

### Step 5: Loss

```python
loss = logZ - gold_score
```

Average across batch.

### Step 6: Viterbi

Implement segmental Viterbi with backpointers.

Return predicted spans:

```text
[(s, e, PEPTIDE), (s, e, PROPEP), ...]
```

Use the same span scoring and transition constraints as training.

### Step 7: Integrate with train loop

Add CLI args:

```text
--model lstmcnn_segmental_crf
--max_seg_len 50
--phase_basis_count 5
--phase_basis_sigma 0.20
--relative_score_scale 0.5 or 1.0
--use_phase_aux_loss false/true
--phase_aux_lambda 0.01
```

Recommended first hyperparameters:

```text
max_seg_len = 50
phase_basis_count = 5
phase_basis_sigma = 0.20
relative_score_scale = 0.5
use_phase_aux_loss = false
```

---

## 19. Testing checklist

### 19.1 Shape tests

Check:

```text
coarse_logits: [B, L, 3]
phase_logits:  [B, L, 2, K]
```

### 19.2 Tiny synthetic sequence

Use a sequence of length 4 or 5 with manually assigned logits.

Verify:

```text
logZ >= gold_score
nll >= 0
```

### 19.3 Brute force comparison

For very small `L <= 5` and `max_seg_len <= 3`, enumerate all possible segmentations by brute force.

Compare:

```text
DP logZ == brute_force logsumexp
DP Viterbi == brute_force max
```

This is the most important correctness test.

### 19.4 Edge cases

Test sequences where:

```text
1. The sequence starts with PEPTIDE.
2. The sequence ends with PEPTIDE.
3. The sequence starts with PROPEP.
4. The sequence ends with PROPEP.
5. There is no None after the final segment.
6. There are no PEPTIDE/PROPEP segments at all.
7. Segment length is 1.
8. Segment length is max_seg_len.
9. A gold segment exceeds max_seg_len -> assert/log/fail clearly.
```

### 19.5 Consistency with old baseline

If `relative_score_scale = 0` and length priors/transitions are neutral, the model should behave similarly to a segment-level version of the coarse emission model:

```text
PEPTIDE span score = sum peptide logits
PROPEP span score = sum propep logits
None score = sum none logits
```

This is a useful debugging mode.

---

## 20. Expected benefits

The current CRF has latent segment states but receives identical emissions for all states of the same class.

The proposed segmental CRF instead scores each candidate segment using its known length. This allows relative-position evidence to be evaluated consistently for variable-length peptides/propeptides.

Expected advantages:

```text
1. No absolute interpretation of Pep1..Pep50 is required.
2. Segment length is known when positional compatibility is evaluated.
3. The model scores full PEPTIDE/PROPEP hypotheses, not only local residue states.
4. The relative-position signal participates directly in training and decoding.
5. Sequence-final segments are handled naturally; no exit transition is required to add a segment bonus.
```

---

## 21. Suggested name for the method

Possible names:

```text
Length-aware Segmental CRF
Relative-position Segmental CRF
Phase-aware Segmental CRF
Length-aware Relative Phase CRF
```

Recommended working name:

```text
Relative-position Segmental CRF
```

Short description:

```text
A segmental CRF that scores candidate PEPTIDE/PROPEP spans using residue-level compatibility with relative position inside the candidate span.
```
