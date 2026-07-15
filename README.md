# Bonsai: Predictive Hebbian Oscillatory Networks

**Bonsai** is a lightweight, biologically inspired framework for modeling neural computation using **phase-coupled oscillators**. It implements both traditional **Hebbian Kuramoto synchronization** and a novel **Predictive Hebbian model** capable of emergent global coherence, pattern completion, and noise robustness.

# Bonsai: Predictive Hebbian Oscillatory Networks

**Bonsai** is a research playground for phase-coupled Kuramoto oscillator
networks with Hebbian (adaptive-weight) coupling, including a hierarchical
extension that adds predictive coding.

**Honest framing:** `HebbianKuramotoOperator` began as a direct implementation
of Bronski et al. (2017), "The stability of fixed points for a Kuramoto model
with Hebbian interactions" (Chaos 27, 053110) -- a pure synchronization/
stability theory, not a classification architecture. `PredictiveHebbianOperator`
extends it with a multi-layer hierarchy and Friston-style predictive coding.
Character-recognition tests were built to explore what that extension does,
and it turned out to be decent at it -- but that emerged from exploration, not
from an original design goal. Keep that in mind when reading the results
below.

## Verified against the underlying theory

`HebbianKuramotoOperator` is checked directly against Bronski et al.'s
published equations and stability theorem (not just informal coherence
thresholds) in `tests/test_hebbian_kuramoto_bronski.py`:
- Its update rule is checked against the paper's equations directly (finite-
  step comparison).
- Its stability behavior is checked against an analytically-derived,
  independently-verifiable fixed-point pair (`maths.graphs.GraphLaplacian.
  from_bronski_stability_matrix`, implementing the paper's Schur-complement
  stability criterion), cross-checked against real simulated recovery/
  divergence from a small perturbation.

A real, previously-undiscovered bug was found and fixed this way: the
coupling term's sign was backwards relative to the paper (silently turning
attractive coupling into repulsive coupling for positive weights) in *three*
places (`models/hebbian/minimalist.py`, `HebbianKuramotoOperator`, and
`PredictiveHebbianOperator`'s within-layer term). It was invisible at exact
synchrony (`sin(0)=0` regardless of sign) and only surfaced once a test
introduced real phase spread. See `docs/` changelogs for the full account.

## Key Features

- **Oscillator-Based Computation** -- Kuramoto-style phase oscillators for
  dynamic, time-continuous representations, with adaptive (Hebbian) coupling
  weights.
- **Hebbian & Predictive Dynamics** -- `HebbianKuramotoOperator` (flat,
  all-to-all, theoretically grounded) vs. `PredictiveHebbianOperator`
  (hierarchical, with between-layer predictive coding on top of within-layer
  Hebbian-Kuramoto coupling).
- **Minimal dependencies, fast execution** -- single CPU core, well under
  50MB RAM, milliseconds-to-seconds per run.

## Verified results (character classification, 7-character alphabet)

Measured via a windowed-average phase readout + nearest-centroid classifier
(`tests/learning/utils/readout.py`; see `tests/test_predictive_hebbian_classification.py`
for the corresponding test):

| Model | Clean accuracy | Noise (30%) | Occlusion (30%) |
|---|---|---|---|
| Hebbian Kuramoto | ~0.14-0.20 (chance is 1/7 ~ 0.14) | ~chance | ~chance |
| Predictive Hebbian | **0.83-0.98** | **0.34-0.66** | **0.69-0.83** |

**Predictive Hebbian genuinely classifies well above chance; Hebbian Kuramoto
does not, under this readout.** This isn't (as far as we've been able to
determine) a remaining bug -- see "Known limitations" below for the actual
mechanistic explanation, which is genuinely interesting: it comes down to
Hebbian's input encoding being an open-loop bias to phase *velocity* (a
drifting signal, sensitive to initial conditions) versus Predictive's
closed-loop correction toward a fixed target *phase* (a stable, reproducible
signal). Occlusion degrading accuracy less than equivalent noise is a real,
mechanistically-sensible finding (occlusion cleanly zeroes sensory error for
missing pixels, allowing pattern completion; noise actively injects
contradictory error) and is protected as a regression test.

Numbers above are ranges, not single points -- there's real run-to-run
variance from random initialization; see the test file for the exact
(evidence-based, not arbitrary) thresholds used.

## Known limitations

- **`HebbianKuramotoOperator` is not currently a good character classifier**
  (see above) -- it's a theoretically well-grounded synchronization model,
  not one built for pattern recognition. Investigated in depth; not yet
  resolved, and may not be a "bug" so much as a structural property of
  open-loop vs. closed-loop input encoding.
- **`PredictiveHebbianOperator` does not converge to a fixed point** in
  general -- it settles into a small, genuine limit-cycle-like oscillation
  rather than a stationary state (this is why the readout uses a windowed
  average rather than a final-state snapshot). Its amplitude dropped
  substantially (~1-2 orders of magnitude) once the coupling sign bug above
  was fixed, but a small residual oscillation persists; not fully explained.
- **`models/akorn/`** (an experimental AKOrN-inspired variant) is currently
  unmaintained and its tests are explicitly skipped (`tests/test_akorne_deluxe.py`)
  -- not being worked on at this time.
- The 7-character alphabet used for the results above is small and
  synthetic (hand-drawn 8x12 bitmaps). It has not been validated against a
  real, standard benchmark -- see Roadmap.

## Getting Started

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
git clone https://github.com/danbarua/bonsai
cd bonsai
uv sync
uv run pytest tests/
uv run python tests/learning/benchmark_character_processing.py
```

## Test suite

- `tests/test_hebbian_kuramoto_bronski.py` -- verification against Bronski
  et al.'s equations and stability theorem.
- `tests/test_hebian_kuramoto.py`, `tests/test_predictive_hebbian_*.py` --
  model mechanics (initialization, edge cases, learning dynamics).
- `tests/test_predictive_hebbian_classification.py` -- real classification-
  accuracy tests for Predictive Hebbian (evidence-based thresholds, not
  arbitrary ones).
- `tests/learning/hebbian/`, `tests/learning/predictive/` -- character-
  processing test suites (shared base class + utilities in
  `tests/learning/utils/`).
- `tests/test_types.py`, `tests/test_akorne_deluxe.py` (skipped) -- domain
  types and the unmaintained AKOrN variant.

## Roadmap

**Validating the classifier against MNIST** is the natural next step to
establish whether the character-recognition result generalizes beyond a
small synthetic alphabet, or is an artifact of a small, easy, hand-designed
character set. See open discussion in project changelogs for what that
would actually require (data pipeline, readout generalization, compute
budget, proper train/test methodology) -- non-trivial, not yet started.

Also open: re-examine `PredictiveHebbianOperator`'s residual limit-cycle
behavior; investigate whether `models/akorn/` shares the coupling-sign bug
pattern found in the other two models.

## Citation & Attribution

- Bronski, J., He, W., Li, Q., Liu, Y., Sponseller, T., Wolbert, S. (2017).
  "The stability of fixed points for a Kuramoto model with Hebbian
  interactions." *Chaos* 27, 053110. [arXiv:1611.09941](https://arxiv.org/abs/1611.09941)
- Predictive coding concepts following Karl Friston's work in theoretical
  neuroscience.

## License

Bonsai is licensed under the MIT License.

