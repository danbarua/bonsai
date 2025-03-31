# Bonsai: Predictive Hebbian Oscillatory Networks

**Bonsai** is a lightweight, biologically inspired framework for modeling neural computation using **phase-coupled oscillators**. It implements both traditional **Hebbian Kuramoto synchronization** and a novel **Predictive Hebbian model** capable of emergent global coherence, pattern completion, and noise robustness.

## Key Features

- **Oscillator-Based Computation**  
  Leverages Kuramoto-style phase oscillators for dynamic, time-continuous representations.

- **Hebbian & Predictive Dynamics**  
  Compare local Hebbian learning with a predictive phase-based inference mechanism.
Predictive model integrates Friston's Predictive Coding concepts in a multi-layer hiearcarchical Hebbian Kuramoto network.

- **Robust to Occlusion & Noise**  
  Predictive model maintains coherence under missing or ambiguous input.

- **High-Dimensional Phase Analysis**  
  Includes tools for measuring coherence, phase gradients, and principal components.

- ⚡ **Minimal Dependencies, Fast Execution**  
  Designed for clarity and experimentation; easily extendable. and it runs on one CPU core, using upto 35MB of RAM, in milliseconds to seconds.

## Example Use Cases

- Character recognition with partial occlusion  
- Phase-based memory and pattern reconstruction  
- Comparing predictive inference vs. local synchronization

## Benchmarked Results

| Model               | Coherence ↑ | Time (ms) ↓ | Noise Robustness ↑ |
|--------------------|-------------|-------------|---------------------|
| Hebbian Kuramoto   | Low         | ✅ Fast      | ❌ Fragile           |
| Predictive Hebbian | **3x Higher**| ⏳ Slower    | ✅ Robust            |

## Getting Started

```bash
git clone https://github.com/danbarua/bonsai
cd bonsai
python tests/learning/benchmark_character_processing.py
```

See `tests/learning/README.md` for details.

## Visualizations

- Phase distributions
- Coherence heatmaps
- Phase difference maps
- Occlusion recovery tests

## Citation & Attribution

This project draws inspiration from:
- Kuramoto oscillator networks  
- Predictive coding theories in neuroscience  
- Hebbian learning and phase coherence metrics

## License

Bonsai is licensed under the MIT License.

---

**Bonsai** is for researchers exploring phase-based neural dynamics, or engineers prototyping brain-like computation in a transparent, interpretable way.


