from typing import NewType, TypeVar, Generic, Annotated, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
import math
from enum import Enum, auto

# Base dimension types to avoid confusion
class Dimension(Enum):
    TIME = auto()
    FREQUENCY = auto()
    SPACE = auto()
    GRAPH = auto()

# Type variable for shape parameters
N = TypeVar('N', bound=int)  # Number of nodes/data points

# === Frequency Types ===

@dataclass(frozen=True)
class FrequencyHz:
    """Frequency in Hertz (cycles per second)"""
    value: float
    
    def __post_init__(self):
        if self.value < 0:
            raise ValueError(f"Frequency must be non-negative, got {self.value}")
    
    def to_rads(self) -> 'FrequencyRads':
        """Convert frequency from Hz to radians per second"""
        return FrequencyRads(self.value * 2 * math.pi)
    
    def __mul__(self, other: float) -> 'FrequencyHz':
        return FrequencyHz(self.value * other)
    
    def __add__(self, other: 'FrequencyHz') -> 'FrequencyHz':
        return FrequencyHz(self.value + other.value)

@dataclass(frozen=True)
class FrequencyRads:
    """Frequency in radians per second"""
    value: float
    
    def __post_init__(self):
        if self.value < 0:
            raise ValueError(f"Frequency must be non-negative, got {self.value}")
    
    def to_hz(self) -> FrequencyHz:
        """Convert frequency from radians per second to Hz"""
        return FrequencyHz(self.value / (2 * math.pi))
    
    def __mul__(self, other: float) -> 'FrequencyRads':
        return FrequencyRads(self.value * other)
    
    def __add__(self, other: 'FrequencyRads') -> 'FrequencyRads':
        return FrequencyRads(self.value + other.value)

@dataclass(frozen=True)
class FrequencyBand:
    """A range of frequencies, like EEG bands (theta, alpha, beta, etc.)"""
    name: str
    min_freq: FrequencyHz
    max_freq: FrequencyHz
    
    def __post_init__(self):
        if self.min_freq.value >= self.max_freq.value:
            raise ValueError("Minimum frequency must be less than maximum frequency")
    
    def contains(self, freq: FrequencyHz) -> bool:
        """Check if the given frequency is within this band"""
        return self.min_freq.value <= freq.value <= self.max_freq.value
    
    @property
    def center_frequency(self) -> FrequencyHz:
        """Get the center frequency of the band"""
        return FrequencyHz((self.min_freq.value + self.max_freq.value) / 2)

# Common EEG frequency bands
DELTA_BAND = FrequencyBand("Delta", FrequencyHz(0.5), FrequencyHz(4))
THETA_BAND = FrequencyBand("Theta", FrequencyHz(4), FrequencyHz(8))
ALPHA_BAND = FrequencyBand("Alpha", FrequencyHz(8), FrequencyHz(13))
BETA_BAND = FrequencyBand("Beta", FrequencyHz(13), FrequencyHz(30))
GAMMA_BAND = FrequencyBand("Gamma", FrequencyHz(30), FrequencyHz(100))

# === Phase Types ===

@dataclass(frozen=True)
class Phase:
    """Angular phase in radians, constrained to [0, 2π)"""
    value: float
    
    def __post_init__(self):
        # Normalize to [0, 2π)
        object.__setattr__(self, 'value', self.value % (2 * math.pi))
    
    def __add__(self, other: 'Phase') -> 'Phase':
        return Phase(self.value + other.value)
    
    def __sub__(self, other: 'Phase') -> 'Phase':
        return Phase(self.value - other.value)
    
    def circular_distance(self, other: 'Phase') -> float:
        """Compute the shortest angular distance between two phases"""
        diff = abs((self.value - other.value) % (2 * math.pi))
        return min(diff, 2 * math.pi - diff)
    
    @staticmethod
    def from_complex(z: complex) -> 'Phase':
        """Create phase from a complex number"""
        return Phase(math.atan2(z.imag, z.real) % (2 * math.pi))
    
    def to_complex(self) -> complex:
        """Convert to complex number on the unit circle"""
        return complex(math.cos(self.value), math.sin(self.value))

@dataclass
class PhaseVector(Generic[N]):
    """A vector of phases for N oscillators"""
    values: NDArray[np.float64]
    
    def __post_init__(self):
        # Normalize all phases to [0, 2π)
        self.values = self.values % (2 * math.pi)
    
    def __array__(self, dtype=None):
        """NumPy array interface"""
        return np.asarray(self.values, dtype=dtype)
    
    def __getitem__(self, idx) -> Phase:
        """Access individual phases"""
        return Phase(self.values[idx])
    
    def __len__(self) -> int:
        return len(self.values)
    
    @property
    def synchronization_order_parameter(self) -> complex:
        """
        Compute the Kuramoto order parameter r*exp(i*ψ)
        
        Returns a complex number where:
        - Magnitude (r) indicates synchronization (0 = no sync, 1 = perfect sync)
        - Angle (ψ) indicates the mean phase
        """
        return np.mean(np.exp(1j * self.values))
    
    @property
    def synchronization_degree(self) -> float:
        """The degree of synchronization (0 to 1)"""
        return abs(self.synchronization_order_parameter)
    
    @property
    def mean_phase(self) -> Phase:
        """The mean phase of all oscillators"""
        return Phase.from_complex(self.synchronization_order_parameter)

# === Spectral Decomposition Types ===

@dataclass
class FrequencyDomainSignal(Generic[N]):
    """
    Representation of a signal in the frequency domain after FFT
    
    Attributes:
        frequencies: The frequency values corresponding to each component
        amplitudes: Magnitude of each frequency component
        phases: Phase angle of each frequency component
        complex_values: Complex FFT output (alternative to amplitude+phase representation)
        dimension: Domain of the original signal (TIME, SPACE, GRAPH)
    """
    frequencies: NDArray[np.float64]  # Shape: (N//2+1,) for real FFT
    amplitudes: NDArray[np.float64]   # Shape: (N//2+1,)
    phases: NDArray[np.float64]       # Shape: (N//2+1,)
    complex_values: NDArray[np.complex128]  # Shape: (N//2+1,) or (N,) depending on FFT type
    dimension: Dimension
    sampling_rate: Optional[float] = None  # Only applicable for TIME dimension
    
    @classmethod
    def from_time_signal(cls, signal: NDArray[np.float64], sampling_rate: float) -> 'FrequencyDomainSignal':
        """Create frequency domain representation from a time domain signal"""
        # Compute FFT
        complex_fft = np.fft.rfft(signal)
        
        # Get frequency values
        n = len(signal)
        freqs = np.fft.rfftfreq(n, d=1/sampling_rate)
        
        # Compute amplitude and phase
        amplitudes = np.abs(complex_fft)
        phases = np.angle(complex_fft)
        
        return cls(
            frequencies=freqs,
            amplitudes=amplitudes,
            phases=phases,
            complex_values=complex_fft,
            dimension=Dimension.TIME,
            sampling_rate=sampling_rate
        )
    
    @classmethod
    def from_graph_signal(cls, signal: NDArray[np.float64], 
                         laplacian_eigvals: NDArray[np.float64],
                         laplacian_eigvecs: NDArray[np.float64]) -> 'FrequencyDomainSignal':
        """Create frequency domain representation from a graph signal using GFT"""
        # Graph Fourier Transform: project signal onto eigenvectors
        gft_coeffs = laplacian_eigvecs.T @ signal
        
        return cls(
            frequencies=laplacian_eigvals,  # Graph frequencies correspond to eigenvalues
            amplitudes=np.abs(gft_coeffs),
            phases=np.angle(gft_coeffs),
            complex_values=gft_coeffs,
            dimension=Dimension.GRAPH
        )
    
    def band_energy(self, band: FrequencyBand) -> float:
        """Compute energy in the given frequency band"""
        if self.dimension != Dimension.TIME:
            raise ValueError(f"Band energy only applicable for time domain signals, not {self.dimension}")
            
        mask = (self.frequencies >= band.min_freq.value) & (self.frequencies <= band.max_freq.value)
        return np.sum(self.amplitudes[mask]**2)
    
    def dominant_frequency(self) -> FrequencyHz:
        """Find the dominant frequency (highest amplitude)"""
        max_idx = np.argmax(self.amplitudes)
        return FrequencyHz(self.frequencies[max_idx])

@dataclass
class SpectralDecomposition(Generic[N]):
    """
    Complete spectral decomposition of a system, including eigenvectors and eigenvalues
    
    This represents a decomposition where a system is broken down into its fundamental
    modes characterized by eigenvalues (frequencies) and eigenvectors (mode shapes).
    """
    eigenvalues: NDArray[np.float64]   # Shape: (N,) - sorted in ascending order
    eigenvectors: NDArray[np.float64]  # Shape: (N, N) - columns are eigenvectors  
    dimension: Dimension
    
    def __post_init__(self):
        n_evals = len(self.eigenvalues)
        n_rows, n_cols = self.eigenvectors.shape
        
        if n_rows != n_cols or n_cols != n_evals:
            raise ValueError(f"Shape mismatch: {n_evals} eigenvalues but eigenvectors shape is {self.eigenvectors.shape}")
    
    @property
    def spectral_gap(self) -> float:
        """
        The spectral gap (difference between first non-zero and zero eigenvalues)
        
        In graph theory, larger spectral gaps indicate better connectivity.
        """
        # Find first non-zero eigenvalue (allowing for numerical precision issues)
        non_zero_idx = np.where(np.abs(self.eigenvalues) > 1e-10)[0][0]
        return self.eigenvalues[non_zero_idx]
    
    def project_signal(self, signal: NDArray[np.float64]) -> FrequencyDomainSignal:
        """Project a signal onto the eigenbasis"""
        if len(signal) != len(self.eigenvalues):
            raise ValueError(f"Signal length {len(signal)} doesn't match decomposition size {len(self.eigenvalues)}")
        
        # Project signal onto eigenvectors
        coefficients = self.eigenvectors.T @ signal
        
        return FrequencyDomainSignal(
            frequencies=self.eigenvalues,
            amplitudes=np.abs(coefficients),
            phases=np.angle(coefficients.astype(complex)),
            complex_values=coefficients.astype(complex),
            dimension=self.dimension
        )
    
    def reconstruct_signal(self, coefficients: NDArray[np.complex128]) -> NDArray[np.float64]:
        """Reconstruct a signal from its spectral coefficients"""
        return np.real(self.eigenvectors @ coefficients)
    
    def filter_signal(self, signal: NDArray[np.float64], cutoff_idx: int) -> NDArray[np.float64]:
        """
        Filter a signal by keeping only components up to cutoff_idx
        
        This implements spectral filtering where:
        - Lower indices = "aligned" components (smooth, global patterns)
        - Higher indices = "liberal" components (detailed, local patterns)
        """
        # Project signal onto eigenbasis
        coeffs = self.eigenvectors.T @ signal
        
        # Apply filter (zero out high-frequency components)
        filtered_coeffs = coeffs.copy()
        filtered_coeffs[cutoff_idx:] = 0
        
        # Reconstruct filtered signal
        return self.eigenvectors @ filtered_coeffs

# === Graph Laplacian Types ===

@dataclass
class GraphLaplacian(Generic[N]):
    """
    Graph Laplacian matrix representation with spectral methods
    
    The Laplacian matrix L = D - A where:
    - D is the degree matrix (diagonal matrix with node degrees)
    - A is the adjacency matrix
    
    Properties:
    - Symmetric for undirected graphs
    - Positive semi-definite
    - Has at least one eigenvalue of 0 (for connected graphs, exactly one is 0)
    - Number of zero eigenvalues equals number of connected components
    """
    matrix: NDArray[np.float64]  # Shape: (N, N)
    is_normalized: bool = False
    
    def __post_init__(self):
        # Verify matrix is square
        n_rows, n_cols = self.matrix.shape
        if n_rows != n_cols:
            raise ValueError(f"Laplacian must be square, got shape {self.matrix.shape}")
        
        # Verify Laplacian properties
        if not np.allclose(self.matrix, self.matrix.T):
            raise ValueError("Laplacian must be symmetric")
        
        # Row sums should be approximately zero for Laplacian
        row_sums = np.abs(np.sum(self.matrix, axis=1))
        if not np.allclose(row_sums, 0, atol=1e-10):
            raise ValueError("Laplacian matrix rows must sum to zero")
    
    @classmethod
    def from_adjacency(cls, adjacency: NDArray[np.float64]) -> 'GraphLaplacian':
        """Create Laplacian from adjacency matrix"""
        # Compute degree matrix (diagonal matrix with row sums of adjacency)
        degrees = np.sum(adjacency, axis=1)
        degree_matrix = np.diag(degrees)
        
        # L = D - A
        laplacian = degree_matrix - adjacency
        
        return cls(matrix=laplacian, is_normalized=False)
    
    @classmethod
    def from_adjacency_normalized(cls, adjacency: NDArray[np.float64]) -> 'GraphLaplacian':
        """Create normalized Laplacian from adjacency matrix"""
        # Compute degree matrix and its inverse square root
        degrees = np.sum(adjacency, axis=1)
        d_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degrees, 1e-10)))
        
        # Standard Laplacian
        degree_matrix = np.diag(degrees)
        laplacian = degree_matrix - adjacency
        
        # Normalized Laplacian: L_norm = D^(-1/2) L D^(-1/2)
        normalized_laplacian = d_inv_sqrt @ laplacian @ d_inv_sqrt
        
        return cls(matrix=normalized_laplacian, is_normalized=True)
    
    def spectral_decomposition(self) -> SpectralDecomposition:
        """Compute the spectral decomposition (eigenvalues and eigenvectors)"""
        # For symmetric matrices, eigenvalues are real
        eigenvalues, eigenvectors = np.linalg.eigh(self.matrix)
        
        # Sort by eigenvalues (smallest first, typically 0 is first)
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        return SpectralDecomposition(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            dimension=Dimension.GRAPH
        )
    
    @property
    def connected_components(self) -> int:
        """Determine number of connected components in the graph"""
        # Number of zero eigenvalues equals number of connected components
        eigenvalues = np.linalg.eigvalsh(self.matrix)
        return np.sum(np.abs(eigenvalues) < 1e-10)
    
    def apply_gft(self, signal: NDArray[np.float64]) -> FrequencyDomainSignal:
        """Apply Graph Fourier Transform to a signal"""
        # Get spectral decomposition
        decomposition = self.spectral_decomposition()
        
        # Project signal onto eigenvectors
        return decomposition.project_signal(signal)
    
    def filter_signal(self, signal: NDArray[np.float64], cutoff_idx: int) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Filter a signal into aligned (low-frequency) and liberal (high-frequency) components
        
        Args:
            signal: Graph signal to filter
            cutoff_idx: Index to split aligned and liberal components
            
        Returns:
            Tuple of (aligned_component, liberal_component)
        """
        # Get spectral decomposition
        decomposition = self.spectral_decomposition()
        
        # Project signal
        coeffs = decomposition.eigenvectors.T @ signal
        
        # Create aligned and liberal masks
        aligned_mask = np.zeros_like(coeffs)
        aligned_mask[:cutoff_idx] = 1
        
        liberal_mask = np.zeros_like(coeffs)
        liberal_mask[cutoff_idx:] = 1
        
        # Create components
        aligned_coeffs = coeffs * aligned_mask
        liberal_coeffs = coeffs * liberal_mask
        
        # Reconstruct components
        aligned_signal = decomposition.eigenvectors @ aligned_coeffs
        liberal_signal = decomposition.eigenvectors @ liberal_coeffs
        
        return aligned_signal, liberal_signal