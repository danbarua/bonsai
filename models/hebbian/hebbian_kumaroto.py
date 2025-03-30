import numpy as np
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, TypeVar, Generic
from numpy.typing import NDArray
from beartype import beartype
from dynamics.oscillators import StateMutation, LayeredOscillatorState

@dataclass
class HebbianKuramotoOperator(StateMutation[LayeredOscillatorState]):
    """
    Implements Kuramoto oscillators with Hebbian plasticity in coupling weights.
    Based on the model from Bronski et al. (2017).
    """
    dt: float = 0.1                # Time step
    mu: float = 0.01               # Hebbian learning rate
    alpha: float = 0.1             # Coupling decay rate
    init_weights: Optional[list[NDArray[np.float64]]] = None  # Initial coupling weights
    weights: list[NDArray[np.float64]] = field(default_factory=list)
    last_delta: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.init_weights is not None:
            self.weights = [w.copy() for w in self.init_weights]
    
    def apply(self, state: LayeredOscillatorState) -> LayeredOscillatorState:
        new_state = state.copy()
        layer_count = state.num_layers
        
        # Initialize weights if not already set
        if not self.weights and layer_count > 0:
            self.weights = []
            for i in range(layer_count):
                shape = state.phases[i].shape
                n_oscillators = np.prod(shape)
                # Create a matrix for each layer's internal coupling
                w = np.random.normal(0, 0.01, (n_oscillators, n_oscillators))
                self.weights.append(w)
        
        # Phase update for all layers
        phase_updates = []
        
        for i in range(layer_count):
            shape = state.phases[i].shape
            n_oscillators = np.prod(shape)
            
            # Flatten phases for matrix operations
            phases_flat = state.phases[i].flatten()
            
            # Compute phase differences matrix (θj - θi)
            phase_diffs = phases_flat[:, np.newaxis] - phases_flat[np.newaxis, :]
            
            # Phase update using current weights and sin of phase differences
            sin_diffs = np.sin(phase_diffs)
            
            # For each oscillator, sum weighted influence from all others
            phase_update_flat = np.sum(self.weights[i] * sin_diffs, axis=1)
            
            # Add natural frequencies 
            phase_update_flat += state.frequencies[i].flatten() * 2 * np.pi #* self.dt
            
            # Reshape back to original shape and store
            phase_update = phase_update_flat.reshape(shape)
            phase_updates.append(phase_update)
            
            # Update coupling weights according to Hebbian rule
            cos_diffs = np.cos(phase_diffs)
            weight_updates = self.mu * cos_diffs - self.alpha * self.weights[i]
            self.weights[i] += self.dt * weight_updates
        
        # Apply phase updates
        for i in range(layer_count):
            new_state._phases[i] = (state.phases[i] + self.dt * phase_updates[i]) % (2 * np.pi)
        
        # Calculate metrics for monitoring
        coherence_values = []
        mean_weights = []
        weight_changes = []
        mean_coherence = 0.0
        max_weight = 0.0
        
        for i in range(layer_count):
            # Phase coherence
            z = np.exp(1j * new_state.phases[i].flatten())
            coherence = float(np.abs(np.mean(z)))
            coherence_values.append(coherence)
            
            # Weight statistics
            mean_weight = float(np.mean(self.weights[i]))
            mean_weights.append(mean_weight)

            max_weight = float(np.max([np.max(w) for w in self.weights]))
            mean_coherence = float(np.mean(coherence_values))
        # Store information about this update
        self.last_delta = {
            "type": "hebbian_kuramoto",
            "coherence": coherence_values,
            "mean_coherence": mean_coherence,
            "mean_weights": mean_weights,
            "max_weight": max_weight
        }
        
        return new_state
    
    def get_delta(self) -> dict[str, Any]:
        return self.last_delta
    
    def debug(self) -> None:
        for k, v in self.last_delta.items():
            print(f"DEBUG: {k} = {v}")