from beartype import beartype
import numpy as np
from numpy.typing import NDArray

@beartype
def update_hebbian_kuramoto(phases: NDArray[np.float64], 
                            weights: NDArray[np.float64], 
                            frequencies: NDArray[np.float64], 
                            dt: float=0.1, 
                            learning_rate: float =0.01, 
                            decay: np.float16=0.1) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Simple update function for Kuramoto oscillators with Hebbian plasticity
    
    Parameters:
    - phases: Array of oscillator phases (radians)
    - weights: Coupling weight matrix
    - frequencies: Natural frequencies of oscillators
    - dt: Time step
    - learning_rate: Hebbian learning rate (mu)
    - decay: Weight decay rate (alpha)
    
    Returns:
    - new_phases: Updated phases
    - new_weights: Updated weights
    """
    n_oscillators = len(phases)
    
    # Compute phase differences matrix
    phase_diffs = phases[:, np.newaxis] - phases[np.newaxis, :]
    
    # Kuramoto phase update
    coupling:NDArray[np.float64] = np.sum(weights * np.sin(phase_diffs), axis=1)
    phase_update:NDArray[np.float64] = frequencies + coupling
    new_phases:NDArray[np.float64] = (phases + dt * phase_update) % (2 * np.pi)
    
    # Hebbian weight update
    weight_update:NDArray[np.float64] = learning_rate * np.cos(phase_diffs) - decay * weights
    new_weights:NDArray[np.float64] = weights + dt * weight_update
    
    return new_phases, new_weights