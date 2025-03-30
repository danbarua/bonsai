from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

import numpy as np
from numpy.typing import NDArray

# Define shape parameters as type variables
N = TypeVar('N', bound=int)  # Number of oscillators
D = TypeVar('D', bound=int)  # Dimensionality

# Define array types with explicit shapes
class Array1D(Generic[N]):
    """Array with shape (N,)"""
    data: NDArray[np.float64]
    
    def __init__(self, data: NDArray[np.float64]):
        if data.ndim != 1:
            raise ValueError(f"Expected 1D array, got shape {data.shape}")
        self.data = data

    
    def __array__(self, dtype=None) -> NDArray:
        """Makes the object compatible with numpy functions expecting ndarray"""
        return np.asarray(self.data, dtype=dtype)
    
    def __getitem__(self, key):
        return self.data[key]
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __shape__(self):
        return self.data.shape
        
    # And other relevant methods like __iter__, shape, dtype properties
    
    @property
    def shape(self) -> tuple[int]:
        return self.data.shape
    
    @property
    def size(self) -> int:
        return self.data.size
        
class Matrix(Generic[N]):
    """Matrix with shape (N, N)"""
    data: NDArray[np.float64]
    
    def __init__(self, data: NDArray[np.float64]):
        if data.ndim != 2 or data.shape[0] != data.shape[1]:
            raise ValueError(f"Expected square matrix, got shape {data.shape}")
        self.data = data
    
    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape

# Then in your oscillator code:
@dataclass
class OscillatorState(Generic[N]):
    """The current state of N oscillators in the network"""
    phases: Array1D[N]  
    amplitudes: Array1D[N]
    
    def __post_init__(self):
        if self.phases.shape != self.amplitudes.shape:
            raise ValueError("Phases and amplitudes must have same shape")