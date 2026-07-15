# Bonsai MNIST — Stage 0 / Stage 0.5, ready to run

Three files, all verified correct (round-tripped the IDX parser against a
synthetic file with known content; ran both baseline scripts end-to-end
against a synthetic MNIST-shaped dataset with genuine class structure and
got 100% accuracy on the easy synthetic case, as expected).

## Why this couldn't be finished in this conversation

The network egress allowlist for this sandbox is fixed at the start of a
conversation (baked into the system prompt), so adding
`ossci-datasets.s3.amazonaws.com` mid-conversation didn't take effect here.
Confirmed by trying the download: got an explicit
"Host not in allowlist: ossci-datasets.s3.amazonaws.com" denial, not a
network timeout.

## To pick this up in a new conversation

1. Start a new chat (needed for the updated allowlist to apply).
2. Upload these three files, or just say "continue the Bonsai MNIST Stage 0
   work" and paste this file's content for context.
3. Download the four MNIST files:
   ```bash
   for f in train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz \
            t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz; do
     curl -sL -o "$f" "https://ossci-datasets.s3.amazonaws.com/mnist/$f"
   done
   ```
4. Sanity check the download actually worked (files should be several MB,
   not ~100 bytes -- that was the tell here that it had failed):
   ```bash
   ls -la *.gz
   python3 mnist_loader.py   # should print real shapes: (60000, 28, 28) etc.
   ```
5. Run both stages:
   ```bash
   python3 stage0_raw_pixel_baseline.py .
   python3 stage0_5_direct_encoding_baseline.py .
   ```
   `stage0_5` takes an optional second arg to subsample the training set for
   a faster first pass, e.g. `python3 stage0_5_direct_encoding_baseline.py . 5000`.

## Files

- `mnist_loader.py` -- pure-NumPy IDX file parser (no sklearn/torch). Verified
  against a synthetic IDX file with known content (exact shape + byte match),
  and against a deliberately-corrupt magic number (correctly raises).
- `stage0_raw_pixel_baseline.py` -- untrained nearest-centroid on raw pixel
  intensities (normalized to [0,1]). The essential control: any oscillator-
  based nearest-centroid readout needs to beat this.
- `stage0_5_direct_encoding_baseline.py` -- reproduces the notebooks'
  `direct_encode` ablation (pixel -> phase -> [cos,sin]) paired with a
  **trained** classifier (sklearn `LogisticRegression`, chosen over a
  from-scratch NumPy MLP for a fast first pass -- swap in an MLP later if a
  closer architectural match to the notebooks matters). This is the
  "trained classifier downstream" methodology, as distinct from Stage 0's
  untrained one -- see the conversation for why both matter (they test
  different things: raw representation separability vs. whether a trainable
  model can exploit the encoding).

Both scripts print per-class accuracy breakdowns, since centroid-based and
even trained methods often do notably better on some digits (0, 1) than
others (8, 9, commonly confused) on real MNIST -- worth knowing which
before drawing conclusions from the headline number.
