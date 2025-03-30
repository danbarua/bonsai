#!/usr/bin/env python
import numpy as np
from maths import THETA_BAND, FrequencyDomainSignal, FrequencyHz, GraphLaplacian, Phase

# === Frequency and Phase Examples ===
# Create frequencies in different units
f_hz = FrequencyHz(8.5)
f_rad = f_hz.to_rads()
print(f"Frequency: {f_hz.value} Hz = {f_rad.value} rad/s")

# Check if a frequency is in a band
if THETA_BAND.contains(f_hz):
    print(f"{f_hz.value} Hz is in the Theta band")
else:
    print(f"{f_hz.value} Hz is in the Alpha band")

# Work with phases
p1 = Phase(0.5)
p2 = Phase(6.0)
distance = p1.circular_distance(p2)
print(f"Circular distance between phases: {distance}")

# === Spectral Analysis Examples ===
# Time series FFT example
time_signal = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 1000)) + 0.5 * np.sin(2 * np.pi * 25 * np.linspace(0, 1, 1000))
freq_domain = FrequencyDomainSignal.from_time_signal(time_signal, sampling_rate=1000)
dominant = freq_domain.dominant_frequency()
print(f"Dominant frequency: {dominant.value} Hz")

# === Graph Signal Processing Examples ===
# Create a simple ring graph adjacency matrix
N = 10
ring_adj = np.zeros((N, N))
for i in range(N):
    ring_adj[i, (i+1)%N] = 1
    ring_adj[(i+1)%N, i] = 1

# Create Laplacian
laplacian = GraphLaplacian.from_adjacency(ring_adj)
decomp = laplacian.spectral_decomposition()

# Create a signal on the graph (e.g., a smooth function of node position)
node_positions = np.linspace(0, 2*np.pi, N, endpoint=False)
graph_signal = np.sin(node_positions)

# Apply GFT
freq_domain = laplacian.apply_gft(graph_signal)
print(f"Graph signal energy in first 3 components: {np.sum(freq_domain.amplitudes[:3]**2)}")

# Split into aligned and liberal components
aligned, liberal = laplacian.filter_signal(graph_signal, cutoff_idx=3)
print(f"Aligned energy: {np.sum(aligned**2)}, Liberal energy: {np.sum(liberal**2)}")