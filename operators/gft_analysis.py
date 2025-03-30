from typing import Any
import numpy as np
from maths.spectral import SpectralDecomposition, FrequencyDomainSignal

from dynamics import LayeredOscillatorState

def analyze_gft_dynamics(self, state: LayeredOscillatorState) -> dict[str, Any]:
    """
    Analyze network dynamics using Graph Fouriter Transform principles
    ---------------------------------------------
    This method directly integrates with GFT approach by:

    Using the Hebbian weight matrices as the connectivity for GFT analysis
    Analyzing how phase patterns project onto the graph's eigenmodes
    Distinguishing between aligned (low-frequency) and liberal (high-frequency) energy components
    Tracking spectral gap as a measure of network modularity
    """
    gft_metrics = []
    
    for i in range(state.num_layers):
        # Use within-layer weights as connectivity for GFT
        A = self.within_layer_weights[i] # TODO: Not actually implemented yet
        
        # Compute normalized graph Laplacian
        D = np.diag(np.sum(A, axis=1))
        L = D - A
        D_sqrt_inv = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-10))
        L_norm = D_sqrt_inv @ L @ D_sqrt_inv
        
        # Compute eigendecomposition (first k eigenvectors)
        k = min(20, L_norm.shape[0]-1)
        try:
            eigvals, eigvecs = np.linalg.eigh(L_norm)
            idx = np.argsort(eigvals)
            eigvals = eigvals[idx[:k]]
            eigvecs = eigvecs[:, idx[:k]]
            
            # Project phase pattern onto GFT basis
            phase_pattern = np.exp(1j * state.phases[i].flatten())
            gft_coeffs = eigvecs.T @ phase_pattern
            
            # Compute aligned vs liberal energy
            cutoff = 5  # First 5 components as "aligned"
            aligned_energy = np.sum(np.abs(gft_coeffs[:cutoff])**2)
            liberal_energy = np.sum(np.abs(gft_coeffs[cutoff:])**2)
            
            gft_metrics.append({
                "layer": i,
                "spectral_gap": float(eigvals[1] - eigvals[0]),
                "aligned_energy": float(aligned_energy),
                "liberal_energy": float(liberal_energy),
                "energy_ratio": float(aligned_energy / (liberal_energy + 1e-10)),
                "dominant_frequency": int(np.argmax(np.abs(gft_coeffs)))
            })
        except:
            gft_metrics.append({
                "layer": i,
                "error": "GFT computation failed"
            })
    
    return {"gft_analysis": gft_metrics}