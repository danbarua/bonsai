import numpy as np
from numpy.typing import NDArray

def kuramoto_update(x:NDArray[np.float64], 
                    omega:NDArray[np.float64], 
                    c:NDArray[np.float64], 
                    J:NDArray[np.float64],  
                    dt:float) -> NDArray[np.float64]:
    """
    Updates the state of Kuramoto oscillators.

    Args:
        x (np.ndarray): Oscillator states (N, dim), where N is the number of oscillators
                         and dim is the dimension of each oscillator.
        omega (np.ndarray): Natural frequencies (N, dim, dim) - anti-symmetric matrices.
        c (np.ndarray): Conditional stimuli (N, dim).
        J (np.ndarray): Coupling strengths (N, N, dim, dim).
        dt (float): Time step.

    Returns:
        np.ndarray: Updated oscillator states (N, dim).
    """
    N = x.shape[0]  # Number of oscillators
    dim = x.shape[1]  # Dimension of each oscillator

    # Natural frequency term ($\Omega_i \mathbf{x}_i$) 
    # np.einsum for efficient matrix multiplication across all oscillators.
    natural_frequency:float = np.einsum('nij,nj->ni', omega, x) # (N, dim)

    # Interaction term
    interaction = c.copy()  # Start with conditional stimuli
    for i in range(N):
        influence = np.zeros(dim)
        for j in range(N):
            influence += J[i, j] @ x[j]  # Matrix multiplication
        interaction[i] += influence

    # Projection onto the tangent space
    # It subtracts the component of the interaction vector 
    # that is parallel to the oscillator's current state (x), 
    # leaving only the component that is tangent to the sphere.
    projection = interaction - np.sum(interaction * x, axis=1, keepdims=True) * x

    # Update oscillator states
    # Calculates the time derivative of the oscillator state.
    x_dot = natural_frequency + projection

    #  Updates the oscillator state using a simple Euler integration step.
    x_new = x + dt * x_dot

    # Normalize to stay on the sphere
    x_new = x_new / np.linalg.norm(x_new, axis=1, keepdims=True)

    return x_new