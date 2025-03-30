from beartype import beartype
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from numpy.typing import NDArray
from dynamics import StateMutation, LayeredOscillatorState

@dataclass
@beartype
class AkornHebbianKuramotoOperator(StateMutation[LayeredOscillatorState]):
    """
    Implements Hebbian-Kuramoto dynamics using the theoretical connection
    to standard Kuramoto models for enhanced stability analysis.
    """
    dt: float = 0.1
    alpha: float = 0.1  # Weight decay parameter
    mu: float = 0.01    # Learning rate
    oscillator_dim: int = 4 # ADDED: Dimensionality of oscillators
    # natural_frequencies: list[NDArray[np.float64]] = field(default_factory=list) # ADDED: Learnable freqs
    weights: list[NDArray[np.float64]] = field(default_factory=list)
    equivalent_kuramoto_states: list[NDArray[np.float64]] = field(default_factory=list)
    last_delta: dict[str, Any] = field(default_factory=dict)
    
    def apply(self, state: LayeredOscillatorState) -> LayeredOscillatorState:
        new_state = state.copy()
        layer_count = state.num_layers
        
        # Initialize if needed
        if not self.weights:
            self.weights = []
            self.equivalent_kuramoto_states = []
            # self.natural_frequencies = [] # ADDED
            for i in range(layer_count):
                shape = state.phases[i].shape
                n_oscillators = np.prod(shape)
                # self.natural_frequencies.append(np.random.normal(0, 0.01, (n_oscillators, self.oscillator_dim, self.oscillator_dim))) # ADDED
                self.weights.append(np.ones((n_oscillators, n_oscillators)) / (2 * self.alpha))
                # MODIFIED: Initialize phases as multi-dimensional vectors
                self.equivalent_kuramoto_states.append(np.random.rand(n_oscillators, self.oscillator_dim))
                self.equivalent_kuramoto_states[i] /= np.linalg.norm(self.equivalent_kuramoto_states[i], axis=1, keepdims=True) # Normalize
                new_state._phases[i] = np.random.rand(n_oscillators, self.oscillator_dim)
                new_state._phases[i] /= np.linalg.norm(new_state._phases[i], axis=1, keepdims=True) # Normalize
                new_state._phases[i] = new_state._phases[i].reshape(shape + (self.oscillator_dim,))
                self.equivalent_kuramoto_states[i] = self.equivalent_kuramoto_states[i].reshape(shape + (self.oscillator_dim,))
        
        # For each layer, update both actual state and equivalent Kuramoto state
        for i in range(layer_count):
            # Get flattened phases
            phases_flat = state.phases[i].reshape(-1, self.oscillator_dim)
            equiv_phases_flat = self.equivalent_kuramoto_states[i].reshape(-1, self.oscillator_dim)
            shape = state.phases[i].shape[:-1] # Shape without oscillator_dim
            n_oscillators = phases_flat.shape[0]
            
            # Update equivalent Kuramoto state (with fixed weights = 1/(2α))
            # This gives us stability information
            equiv_diffs = equiv_phases_flat[:, np.newaxis, :] - equiv_phases_flat[np.newaxis, :, :] # (N, N, dim)
            equiv_update = state.frequencies[i].flatten() * 2 * np.pi #+ self.natural_frequencies[i] @ equiv_phases_flat # (N, dim)
            equiv_update += np.sum(np.sin(equiv_diffs) / (2 * self.alpha), axis=1) # (N, dim)
            equiv_new = (equiv_phases_flat + self.dt * equiv_update) # (N, dim)
            equiv_new /= np.linalg.norm(equiv_new, axis=1, keepdims=True) # Normalize
            
            # Update actual Hebbian-Kuramoto state
            # The fixed point is at half the phase angle
            actual_update = state.frequencies[i].flatten() * 2 * np.pi #+ self.natural_frequencies[i] @ phases_flat # (N, dim)
            
            # Compute phase differences
            phase_diffs = phases_flat[:, np.newaxis, :] - phases_flat[np.newaxis, :, :] # (N, N, dim)
            
            # Get current weights from cosine of phase differences (fixed point relation)
            current_weights = np.cos(np.sum(phase_diffs**2, axis=2)) / self.alpha # (N, N) - cosine of angle
            
            # Use these weights with sin term for phase update
            actual_update += np.sum(current_weights[:, :, np.newaxis] * np.sin(phase_diffs), axis=1) # (N, dim)
            actual_new = (phases_flat + self.dt * actual_update) # (N, dim)
            actual_new /= np.linalg.norm(actual_new, axis=1, keepdims=True) # Normalize
            
            # Store updated states
            self.equivalent_kuramoto_states[i] = equiv_new.reshape(shape + (self.oscillator_dim,))
            new_state._phases[i] = actual_new.reshape(shape + (self.oscillator_dim,))
            
            # Update weights according to Hebbian rule
            self.weights[i] = current_weights
        
        # Compute metrics
        coherence_values = []
        for i in range(layer_count):
            # MODIFIED: Calculate coherence using vector states
            z = new_state.phases[i].reshape(-1, self.oscillator_dim)
            coherence = float(np.abs(np.mean(z))) # Simple average - can be improved
            coherence_values.append(coherence)
        
        # Check stability using equivalent Kuramoto dynamics
        stability_metrics = self._analyze_stability(new_state.phases[i])
        
        self.last_delta = {
            "type": "enhanced_hebbian_kuramoto",
            "coherence": coherence_values,
            "mean_coherence": float(np.mean(coherence_values)),
            "stability": stability_metrics
        }
        
        return new_state
    
    def _analyze_stability(self, phases) -> dict[str, Any]:
        """Analyze stability using the equivalent Kuramoto model"""
        # This would implement the stability analysis based on the paper's findings
        # For now just return placeholder metrics
        # MODIFIED: Calculate a more meaningful stability metric
        # Example: Variance of phase differences
        phase_diffs = phases[:, np.newaxis] - phases[np.newaxis, :]
        phase_variance = np.var(phase_diffs)
        return {
            "phase_variance": float(phase_variance)
        }
    
    def get_delta(self) -> dict[str, Any]:
        return self.last_delta