import unittest
import numpy as np
import math
from enum import Enum, auto
from typing import NewType, TypeVar, Generic, Annotated, List, Tuple, Optional
from dataclasses import dataclass
from maths import Dimension, FrequencyBand, FrequencyDomainSignal, FrequencyHz, FrequencyRads, GraphLaplacian, Phase, PhaseVector, SpectralDecomposition


# # === Domain Types (as defined in the note) ===

# # Base dimension types to avoid confusion
# class Dimension(Enum):
#     TIME = auto()
#     FREQUENCY = auto()
#     SPACE = auto()
#     GRAPH = auto()

# # Type variable for shape parameters
# N = TypeVar('N', bound=int)  # Number of nodes/data points

# # === Frequency Types ===

# @dataclass(frozen=True)
# class FrequencyHz:
#     """Frequency in Hertz (cycles per second)"""
#     value: float
    
#     def __post_init__(self):
#         if self.value < 0:
#             raise ValueError(f"Frequency must be non-negative, got {self.value}")
    
#     def to_rads(self) -> 'FrequencyRads':
#         """Convert frequency from Hz to radians per second"""
#         return FrequencyRads(self.value * 2 * math.pi)
    
#     def __mul__(self, other: float) -> 'FrequencyHz':
#         return FrequencyHz(self.value * other)
    
#     def __add__(self, other: 'FrequencyHz') -> 'FrequencyHz':
#         return FrequencyHz(self.value + other.value)

# @dataclass(frozen=True)
# class FrequencyRads:
#     """Frequency in radians per second"""
#     value: float
    
#     def __post_init__(self):
#         if self.value < 0:
#             raise ValueError(f"Frequency must be non-negative, got {self.value}")
    
#     def to_hz(self) -> FrequencyHz:
#         """Convert frequency from radians per second to Hz"""
#         return FrequencyHz(self.value / (2 * math.pi))
    
#     def __mul__(self, other: float) -> 'FrequencyRads':
#         return FrequencyRads(self.value * other)
    
#     def __add__(self, other: 'FrequencyRads') -> 'FrequencyRads':
#         return FrequencyRads(self.value + other.value)

# @dataclass(frozen=True)
# class FrequencyBand:
#     """A range of frequencies, like EEG bands (theta, alpha, beta, etc.)"""
#     name: str
#     min_freq: FrequencyHz
#     max_freq: FrequencyHz
    
#     def __post_init__(self):
#         if self.min_freq.value >= self.max_freq.value:
#             raise ValueError("Minimum frequency must be less than maximum frequency")
    
#     def contains(self, freq: FrequencyHz) -> bool:
#         """Check if the given frequency is within this band"""
#         return self.min_freq.value <= freq.value <= self.max_freq.value
    
#     @property
#     def center_frequency(self) -> FrequencyHz:
#         """Get the center frequency of the band"""
#         return FrequencyHz((self.min_freq.value + self.max_freq.value) / 2)

# # Common EEG frequency bands
# DELTA_BAND = FrequencyBand("Delta", FrequencyHz(0.5), FrequencyHz(4))
# THETA_BAND = FrequencyBand("Theta", FrequencyHz(4), FrequencyHz(8))
# ALPHA_BAND = FrequencyBand("Alpha", FrequencyHz(8), FrequencyHz(13))
# BETA_BAND = FrequencyBand("Beta", FrequencyHz(13), FrequencyHz(30))
# GAMMA_BAND = FrequencyBand("Gamma", FrequencyHz(30), FrequencyHz(100))

# # === Phase Types ===

# @dataclass(frozen=True)
# class Phase:
#     """Angular phase in radians, constrained to [0, 2π)"""
#     value: float
    
#     def __post_init__(self):
#         # Normalize to [0, 2π)
#         object.__setattr__(self, 'value', self.value % (2 * math.pi))
    
#     def __add__(self, other: 'Phase') -> 'Phase':
#         return Phase(self.value + other.value)
    
#     def __sub__(self, other: 'Phase') -> 'Phase':
#         return Phase(self.value - other.value)
    
#     def circular_distance(self, other: 'Phase') -> float:
#         """Compute the shortest angular distance between two phases"""
#         diff = abs((self.value - other.value) % (2 * math.pi))
#         return min(diff, 2 * math.pi - diff)
    
#     @staticmethod
#     def from_complex(z: complex) -> 'Phase':
#         """Create phase from a complex number"""
#         return Phase(math.atan2(z.imag, z.real) % (2 * math.pi))
    
#     def to_complex(self) -> complex:
#         """Convert to complex number on the unit circle"""
#         return complex(math.cos(self.value), math.sin(self.value))

# @dataclass
# class PhaseVector(Generic[N]):
#     """A vector of phases for N oscillators"""
#     values: np.ndarray
    
#     def __post_init__(self):
#         # Normalize all phases to [0, 2π)
#         self.values = self.values % (2 * math.pi)
    
#     def __array__(self, dtype=None):
#         """NumPy array interface"""
#         return np.asarray(self.values, dtype=dtype)
    
#     def __getitem__(self, idx) -> Phase:
#         """Access individual phases"""
#         return Phase(self.values[idx])
    
#     def __len__(self) -> int:
#         return len(self.values)
    
#     @property
#     def synchronization_order_parameter(self) -> complex:
#         """
#         Compute the Kuramoto order parameter r*exp(i*ψ)
        
#         Returns a complex number where:
#         - Magnitude (r) indicates synchronization (0 = no sync, 1 = perfect sync)
#         - Angle (ψ) indicates the mean phase
#         """
#         return np.mean(np.exp(1j * self.values))
    
#     @property
#     def synchronization_degree(self) -> float:
#         """The degree of synchronization (0 to 1)"""
#         return abs(self.synchronization_order_parameter)
    
#     @property
#     def mean_phase(self) -> Phase:
#         """The mean phase of all oscillators"""
#         return Phase.from_complex(self.synchronization_order_parameter)

# # === Spectral Decomposition Types ===

# @dataclass
# class FrequencyDomainSignal(Generic[N]):
#     """
#     Representation of a signal in the frequency domain after FFT
    
#     Attributes:
#         frequencies: The frequency values corresponding to each component
#         amplitudes: Magnitude of each frequency component
#         phases: Phase angle of each frequency component
#         complex_values: Complex FFT output (alternative to amplitude+phase representation)
#         dimension: Domain of the original signal (TIME, SPACE, GRAPH)
#     """
#     frequencies: np.ndarray  # Shape: (N//2+1,) for real FFT
#     amplitudes: np.ndarray   # Shape: (N//2+1,)
#     phases: np.ndarray       # Shape: (N//2+1,)
#     complex_values: np.ndarray  # Shape: (N//2+1,) or (N,) depending on FFT type
#     dimension: Dimension
#     sampling_rate: Optional[float] = None  # Only applicable for TIME dimension
    
#     @classmethod
#     def from_time_signal(cls, signal: np.ndarray, sampling_rate: float) -> 'FrequencyDomainSignal':
#         """Create frequency domain representation from a time domain signal"""
#         # Compute FFT
#         complex_fft = np.fft.rfft(signal)
        
#         # Get frequency values
#         n = len(signal)
#         freqs = np.fft.rfftfreq(n, d=1/sampling_rate)
        
#         # Compute amplitude and phase
#         amplitudes = np.abs(complex_fft)
#         phases = np.angle(complex_fft)
        
#         return cls(
#             frequencies=freqs,
#             amplitudes=amplitudes,
#             phases=phases,
#             complex_values=complex_fft,
#             dimension=Dimension.TIME,
#             sampling_rate=sampling_rate
#         )
    
#     @classmethod
#     def from_graph_signal(cls, signal: np.ndarray, 
#                          laplacian_eigvals: np.ndarray,
#                          laplacian_eigvecs: np.ndarray) -> 'FrequencyDomainSignal':
#         """Create frequency domain representation from a graph signal using GFT"""
#         # Graph Fourier Transform: project signal onto eigenvectors
#         gft_coeffs = laplacian_eigvecs.T @ signal
        
#         return cls(
#             frequencies=laplacian_eigvals,  # Graph frequencies correspond to eigenvalues
#             amplitudes=np.abs(gft_coeffs),
#             phases=np.angle(gft_coeffs),
#             complex_values=gft_coeffs,
#             dimension=Dimension.GRAPH
#         )
    
#     def band_energy(self, band: 'FrequencyBand') -> float:
#         """Compute energy in the given frequency band"""
#         if self.dimension != Dimension.TIME:
#             raise ValueError(f"Band energy only applicable for time domain signals, not {self.dimension}")
            
#         mask = (self.frequencies >= band.min_freq.value) & (self.frequencies <= band.max_freq.value)
#         return np.sum(self.amplitudes[mask]**2)
    
#     def dominant_frequency(self) -> 'FrequencyHz':
#         """Find the dominant frequency (highest amplitude)"""
#         max_idx = np.argmax(self.amplitudes)
#         return FrequencyHz(self.frequencies[max_idx])

# @dataclass
# class SpectralDecomposition(Generic[N]):
#     """
#     Complete spectral decomposition of a system, including eigenvectors and eigenvalues
    
#     This represents a decomposition where a system is broken down into its fundamental
#     modes characterized by eigenvalues (frequencies) and eigenvectors (mode shapes).
#     """
#     eigenvalues: np.ndarray   # Shape: (N,) - sorted in ascending order
#     eigenvectors: np.ndarray  # Shape: (N, N) - columns are eigenvectors  
#     dimension: Dimension
    
#     def __post_init__(self):
#         n_evals = len(self.eigenvalues)
#         n_rows, n_cols = self.eigenvectors.shape
        
#         if n_rows != n_cols or n_cols != n_evals:
#             raise ValueError(f"Shape mismatch: {n_evals} eigenvalues but eigenvectors shape is {self.eigenvectors.shape}")
    
#     @property
#     def spectral_gap(self) -> float:
#         """
#         The spectral gap (difference between first non-zero and zero eigenvalues)
        
#         In graph theory, larger spectral gaps indicate better connectivity.
#         """
#         # Find first non-zero eigenvalue (allowing for numerical precision issues)
#         non_zero_idx = np.where(np.abs(self.eigenvalues) > 1e-10)[0][0]
#         return self.eigenvalues[non_zero_idx]
    
#     def project_signal(self, signal: np.ndarray) -> 'FrequencyDomainSignal':
#         """Project a signal onto the eigenbasis"""
#         if len(signal) != len(self.eigenvalues):
#             raise ValueError(f"Signal length {len(signal)} doesn't match decomposition size {len(self.eigenvalues)}")
        
#         # Project signal onto eigenvectors
#         coefficients = self.eigenvectors.T @ signal
        
#         return FrequencyDomainSignal(
#             frequencies=self.eigenvalues,
#             amplitudes=np.abs(coefficients),
#             phases=np.angle(coefficients.astype(complex)),
#             complex_values=coefficients.astype(complex),
#             dimension=self.dimension
#         )
    
#     def reconstruct_signal(self, coefficients: np.ndarray) -> np.ndarray:
#         """Reconstruct a signal from its spectral coefficients"""
#         return np.real(self.eigenvectors @ coefficients)
    
#     def filter_signal(self, signal: np.ndarray, cutoff_idx: int) -> np.ndarray:
#         """
#         Filter a signal by keeping only components up to cutoff_idx
        
#         This implements spectral filtering where:
#         - Lower indices = "aligned" components (smooth, global patterns)
#         - Higher indices = "liberal" components (detailed, local patterns)
#         """
#         # Project signal onto eigenbasis
#         coeffs = self.eigenvectors.T @ signal
        
#         # Apply filter (zero out high-frequency components)
#         filtered_coeffs = coeffs.copy()
#         filtered_coeffs[cutoff_idx:] = 0
        
#         # Reconstruct filtered signal
#         return self.eigenvectors @ filtered_coeffs

# # === Graph Laplacian Types ===

# @dataclass
# class GraphLaplacian(Generic[N]):
#     """
#     Graph Laplacian matrix representation with spectral methods
    
#     The Laplacian matrix L = D - A where:
#     - D is the degree matrix (diagonal matrix with node degrees)
#     - A is the adjacency matrix
    
#     Properties:
#     - Symmetric for undirected graphs
#     - Positive semi-definite
#     - Has at least one eigenvalue of 0 (for connected graphs, exactly one is 0)
#     - Number of zero eigenvalues equals number of connected components
#     """
#     matrix: np.ndarray  # Shape: (N, N)
#     is_normalized: bool = False
    
#     def __post_init__(self):
#         # Verify matrix is square
#         n_rows, n_cols = self.matrix.shape
#         if n_rows != n_cols:
#             raise ValueError(f"Laplacian must be square, got shape {self.matrix.shape}")
        
#         # Verify Laplacian properties
#         if not np.allclose(self.matrix, self.matrix.T):
#             raise ValueError("Laplacian must be symmetric")
        
#         # Row sums should be approximately zero for Laplacian
#         row_sums = np.abs(np.sum(self.matrix, axis=1))
#         if not np.allclose(row_sums, 0, atol=1e-10):
#             raise ValueError("Laplacian matrix rows must sum to zero")
    
#     @classmethod
#     def from_adjacency(cls, adjacency: np.ndarray) -> 'GraphLaplacian':
#         """Create Laplacian from adjacency matrix"""
#         # Compute degree matrix (diagonal matrix with row sums of adjacency)
#         degrees = np.sum(adjacency, axis=1)
#         degree_matrix = np.diag(degrees)
        
#         # L = D - A
#         laplacian = degree_matrix - adjacency
        
#         return cls(matrix=laplacian, is_normalized=False)
    
#     @classmethod
#     def from_adjacency_normalized(cls, adjacency: np.ndarray) -> 'GraphLaplacian':
#         """Create normalized Laplacian from adjacency matrix"""
#         # Compute degree matrix and its inverse square root
#         degrees = np.sum(adjacency, axis=1)
#         d_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degrees, 1e-10)))
        
#         # Standard Laplacian
#         degree_matrix = np.diag(degrees)
#         laplacian = degree_matrix - adjacency
        
#         # Normalized Laplacian: L_norm = D^(-1/2) L D^(-1/2)
#         normalized_laplacian = d_inv_sqrt @ laplacian @ d_inv_sqrt
        
#         return cls(matrix=normalized_laplacian, is_normalized=True)
    
#     def spectral_decomposition(self) -> 'SpectralDecomposition':
#         """Compute the spectral decomposition (eigenvalues and eigenvectors)"""
#         # For symmetric matrices, eigenvalues are real
#         eigenvalues, eigenvectors = np.linalg.eigh(self.matrix)
        
#         # Sort by eigenvalues (smallest first, typically 0 is first)
#         idx = np.argsort(eigenvalues)
#         eigenvalues = eigenvalues[idx]
#         eigenvectors = eigenvectors[:, idx]
        
#         return SpectralDecomposition(
#             eigenvalues=eigenvalues,
#             eigenvectors=eigenvectors,
#             dimension=Dimension.GRAPH
#         )
    
#     @property
#     def connected_components(self) -> int:
#         """Determine number of connected components in the graph"""
#         # Number of zero eigenvalues equals number of connected components
#         eigenvalues = np.linalg.eigvalsh(self.matrix)
#         return np.sum(np.abs(eigenvalues) < 1e-10)
    
#     def apply_gft(self, signal: np.ndarray) -> 'FrequencyDomainSignal':
#         """Apply Graph Fourier Transform to a signal"""
#         # Get spectral decomposition
#         decomposition = self.spectral_decomposition()
        
#         # Project signal onto eigenvectors
#         return decomposition.project_signal(signal)
    
#     def filter_signal(self, signal: np.ndarray, cutoff_idx: int) -> Tuple[np.ndarray, np.ndarray]:
#         """
#         Filter a signal into aligned (low-frequency) and liberal (high-frequency) components
        
#         Args:
#             signal: Graph signal to filter
#             cutoff_idx: Index to split aligned and liberal components
            
#         Returns:
#             Tuple of (aligned_component, liberal_component)
#         """
#         # Get spectral decomposition
#         decomposition = self.spectral_decomposition()
        
#         # Project signal
#         coeffs = decomposition.eigenvectors.T @ signal
        
#         # Create aligned and liberal masks
#         aligned_mask = np.zeros_like(coeffs)
#         aligned_mask[:cutoff_idx] = 1
        
#         liberal_mask = np.zeros_like(coeffs)
#         liberal_mask[cutoff_idx:] = 1
        
#         # Create components
#         aligned_coeffs = coeffs * aligned_mask
#         liberal_coeffs = coeffs * liberal_mask
        
#         # Reconstruct components
#         aligned_signal = decomposition.eigenvectors @ aligned_coeffs
#         liberal_signal = decomposition.eigenvectors @ liberal_coeffs
        
#         return aligned_signal, liberal_signal

# === Unit Tests ===

class TestDomainTypes(unittest.TestCase):

    def test_frequency_hz_creation(self):
        f = FrequencyHz(50.0)
        self.assertEqual(f.value, 50.0)
        with self.assertRaises(ValueError):
            FrequencyHz(-10)

    def test_frequency_hz_to_rads(self):
        f_hz = FrequencyHz(1)
        f_rads = f_hz.to_rads()
        self.assertAlmostEqual(f_rads.value, 2 * math.pi)

    def test_frequency_hz_operations(self):
        f1 = FrequencyHz(10)
        f2 = FrequencyHz(20)
        self.assertEqual((f1 * 2).value, 20)
        self.assertEqual((f1 + f2).value, 30)

    def test_frequency_rads_creation(self):
        f = FrequencyRads(50.0)
        self.assertEqual(f.value, 50.0)
        with self.assertRaises(ValueError):
            FrequencyRads(-10)

    def test_frequency_rads_to_hz(self):
        f_rads = FrequencyRads(2 * math.pi)
        f_hz = f_rads.to_hz()
        self.assertAlmostEqual(f_hz.value, 1)

    def test_frequency_rads_operations(self):
        f1 = FrequencyRads(10)
        f2 = FrequencyRads(20)
        self.assertEqual((f1 * 2).value, 20)
        self.assertEqual((f1 + f2).value, 30)

    def test_frequency_band_creation(self):
        band = FrequencyBand("Test", FrequencyHz(10), FrequencyHz(20))
        self.assertEqual(band.name, "Test")
        self.assertEqual(band.min_freq.value, 10)
        self.assertEqual(band.max_freq.value, 20)
        with self.assertRaises(ValueError):
            FrequencyBand("Invalid", FrequencyHz(20), FrequencyHz(10))

    def test_frequency_band_contains(self):
        band = FrequencyBand("Test", FrequencyHz(10), FrequencyHz(20))
        self.assertTrue(band.contains(FrequencyHz(15)))
        self.assertFalse(band.contains(FrequencyHz(5)))
        self.assertFalse(band.contains(FrequencyHz(25)))

    def test_frequency_band_center_frequency(self):
        band = FrequencyBand("Test", FrequencyHz(10), FrequencyHz(20))
        self.assertEqual(band.center_frequency.value, 15)

    def test_phase_creation(self):
        p = Phase(math.pi / 2)
        self.assertAlmostEqual(p.value, math.pi / 2)
        # Test normalization
        p2 = Phase(3 * math.pi)
        self.assertAlmostEqual(p2.value, math.pi)

    def test_phase_operations(self):
        p1 = Phase(math.pi / 4)
        p2 = Phase(math.pi / 2)
        self.assertAlmostEqual((p1 + p2).value, 3 * math.pi / 4)
        self.assertAlmostEqual((p2 - p1).value, math.pi / 4)

    def test_phase_circular_distance(self):
        p1 = Phase(0)
        p2 = Phase(math.pi)
        self.assertAlmostEqual(p1.circular_distance(p2), math.pi)
        p3 = Phase(math.pi / 4)
        p4 = Phase(7 * math.pi / 4)
        self.assertAlmostEqual(p3.circular_distance(p4), math.pi / 2)

    def test_phase_from_complex(self):
        z = complex(1, 1)  # 45 degrees
        p = Phase.from_complex(z)
        self.assertAlmostEqual(p.value, math.pi / 4)

    def test_phase_to_complex(self):
        p = Phase(math.pi / 2)
        z = p.to_complex()
        self.assertAlmostEqual(z.real, 0)
        self.assertAlmostEqual(z.imag, 1)

    def test_phase_vector_creation(self):
        phases = np.array([0, math.pi / 2, math.pi])
        pv = PhaseVector(phases)
        np.testing.assert_array_almost_equal(pv.values, phases)

    def test_phase_vector_normalization(self):
        phases = np.array([0, 5 * math.pi / 2, 3 * math.pi])
        pv = PhaseVector(phases)
        np.testing.assert_array_almost_equal(pv.values, [0, math.pi / 2, math.pi])

    def test_phase_vector_getitem(self):
        phases = np.array([0, math.pi / 2, math.pi])
        pv = PhaseVector(phases)
        self.assertAlmostEqual(pv[1].value, math.pi / 2)

    def test_phase_vector_len(self):
        phases = np.array([0, math.pi / 2, math.pi])
        pv = PhaseVector(phases)
        self.assertEqual(len(pv), 3)

    def test_phase_vector_synchronization_order_parameter(self):
        phases = np.array([0, 0, 0])
        pv = PhaseVector(phases)
        r_exp_i_psi = pv.synchronization_order_parameter
        self.assertAlmostEqual(abs(r_exp_i_psi), 1)  # Perfect sync
        self.assertAlmostEqual(np.angle(r_exp_i_psi), 0)

        phases_desync = np.array([0, math.pi, math.pi/2])
        pv_desync = PhaseVector(phases_desync)
        r_exp_i_psi_desync = pv_desync.synchronization_order_parameter
        self.assertLess(abs(r_exp_i_psi_desync), 1)

    def test_phase_vector_synchronization_degree(self):
        phases = np.array([0, 0, 0])
        pv = PhaseVector(phases)
        self.assertAlmostEqual(pv.synchronization_degree, 1)

        phases = np.array([0, math.pi])
        pv = PhaseVector(phases)
        self.assertAlmostEqual(pv.synchronization_degree, 0)

    def test_phase_vector_mean_phase(self):
        phases = np.array([0, math.pi / 2])
        pv = PhaseVector(phases)
        self.assertAlmostEqual(pv.mean_phase.value, math.atan2(1,1))

    def test_frequency_domain_signal_from_time_signal(self):
        # Create a simple sine wave
        sampling_rate = 100
        time = np.linspace(0, 1, sampling_rate, endpoint=False)
        signal = np.sin(2 * np.pi * 5 * time)  # 5 Hz sine wave

        # Create FrequencyDomainSignal
        fds = FrequencyDomainSignal.from_time_signal(signal, sampling_rate)

        self.assertEqual(fds.dimension, Dimension.TIME)
        self.assertEqual(fds.sampling_rate, sampling_rate)
        self.assertEqual(len(fds.frequencies), len(fds.amplitudes))
        self.assertEqual(len(fds.frequencies), len(fds.phases))
        self.assertEqual(len(fds.frequencies), len(fds.complex_values))

        # Check if the dominant frequency is around 5 Hz
        dominant_frequency = fds.dominant_frequency()
        self.assertAlmostEqual(dominant_frequency.value, 5, delta=1)

    def test_frequency_domain_signal_from_graph_signal(self):
        # Create dummy data for graph signal and Laplacian
        num_nodes = 10
        graph_signal = np.random.rand(num_nodes)
        laplacian_eigvals = np.sort(np.random.rand(num_nodes))
        laplacian_eigvecs = np.random.rand(num_nodes, num_nodes)

        # Create FrequencyDomainSignal
        fds = FrequencyDomainSignal.from_graph_signal(graph_signal, laplacian_eigvals, laplacian_eigvecs)

        self.assertEqual(fds.dimension, Dimension.GRAPH)
        self.assertEqual(len(fds.frequencies), num_nodes)
        self.assertEqual(len(fds.amplitudes), num_nodes)
        self.assertEqual(len(fds.phases), num_nodes)
        self.assertEqual(len(fds.complex_values), num_nodes)
        np.testing.assert_array_equal(fds.frequencies, laplacian_eigvals)

    def test_frequency_domain_signal_band_energy(self):
        # Create a FrequencyDomainSignal (example data)
        sampling_rate = 100
        time = np.linspace(0, 1, sampling_rate, endpoint=False)
        signal = np.sin(2 * np.pi * 5 * time) + 0.5 * np.sin(2 * np.pi * 15 * time)
        fds = FrequencyDomainSignal.from_time_signal(signal, sampling_rate)

        # Define a frequency band
        band = FrequencyBand("TestBand", FrequencyHz(4), FrequencyHz(6))

        # Calculate band energy
        energy = fds.band_energy(band)

        # The energy should be greater than 0 since there's a 5 Hz component
        self.assertGreater(energy, 0)

        # Test ValueError for non-time dimension
        fds.dimension = Dimension.GRAPH
        with self.assertRaises(ValueError):
            fds.band_energy(band)

    def test_frequency_domain_signal_dominant_frequency(self):
        # Create a FrequencyDomainSignal (example data)
        sampling_rate = 100
        time = np.linspace(0, 1, sampling_rate, endpoint=False)
        signal = np.sin(2 * np.pi * 5 * time) + 0.5 * np.sin(2 * np.pi * 15 * time)
        fds = FrequencyDomainSignal.from_time_signal(signal, sampling_rate)

        # Find dominant frequency
        dominant_frequency = fds.dominant_frequency()

        # The dominant frequency should be around 5 Hz
        self.assertAlmostEqual(dominant_frequency.value, 5, delta=1)

    def test_spectral_decomposition_creation(self):
        # Create dummy data
        num_nodes = 10
        eigenvalues = np.sort(np.random.rand(num_nodes))
        eigenvectors = np.random.rand(num_nodes, num_nodes)

        # Create SpectralDecomposition
        sd = SpectralDecomposition(eigenvalues, eigenvectors, Dimension.GRAPH)

        self.assertEqual(len(sd.eigenvalues), num_nodes)
        self.assertEqual(sd.eigenvectors.shape, (num_nodes, num_nodes))
        self.assertEqual(sd.dimension, Dimension.GRAPH)

    def test_spectral_decomposition_shape_mismatch(self):
        # Create dummy data with shape mismatch
        num_nodes = 10
        eigenvalues = np.sort(np.random.rand(num_nodes))
        eigenvectors = np.random.rand(num_nodes, num_nodes + 1)  # Shape mismatch

        # Expect ValueError
        with self.assertRaises(ValueError):
            SpectralDecomposition(eigenvalues, eigenvectors, Dimension.GRAPH)

    def test_spectral_decomposition_spectral_gap(self):
        # Create dummy data
        eigenvalues = np.array([0.0, 0.1, 0.5, 1.0])
        eigenvectors = np.random.rand(4, 4)

        # Create SpectralDecomposition
        sd = SpectralDecomposition(eigenvalues, eigenvectors, Dimension.GRAPH)

        # Spectral gap should be 0.1
        self.assertAlmostEqual(sd.spectral_gap, 0.1)

    def test_spectral_decomposition_project_signal(self):
        # Create dummy data
        num_nodes = 10
        eigenvalues = np.sort(np.random.rand(num_nodes))
        eigenvectors = np.random.rand(num_nodes, num_nodes)
        signal = np.random.rand(num_nodes)

        # Create SpectralDecomposition
        sd = SpectralDecomposition(eigenvalues, eigenvectors, Dimension.GRAPH)

        # Project signal
        fds = sd.project_signal(signal)

        self.assertIsInstance(fds, FrequencyDomainSignal)
        self.assertEqual(len(fds.frequencies), num_nodes)

    def test_spectral_decomposition_project_signal_length_mismatch(self):
        # Create dummy data with length mismatch
        num_nodes = 10
        eigenvalues = np.sort(np.random.rand(num_nodes))
        eigenvectors = np.random.rand(num_nodes, num_nodes)
        signal = np.random.rand(num_nodes + 1)  # Length mismatch

        # Create SpectralDecomposition
        sd = SpectralDecomposition(eigenvalues, eigenvectors, Dimension.GRAPH)

        # Expect ValueError
        with self.assertRaises(ValueError):
            sd.project_signal(signal)

    def test_spectral_decomposition_reconstruct_signal(self):
        # Create dummy data
        num_nodes = 10
        eigenvalues = np.sort(np.random.rand(num_nodes))
        eigenvectors = np.random.rand(num_nodes, num_nodes)
        coefficients = np.random.rand(num_nodes)

        # Create SpectralDecomposition
        sd = SpectralDecomposition(eigenvalues, eigenvectors, Dimension.GRAPH)

        # Reconstruct signal
        reconstructed_signal = sd.reconstruct_signal(coefficients)

        self.assertEqual(len(reconstructed_signal), num_nodes)

    def test_spectral_decomposition_filter_signal(self):
        # Create dummy data
        num_nodes = 10
        eigenvalues = np.sort(np.random.rand(num_nodes))
        eigenvectors = np.random.rand(num_nodes, num_nodes)
        signal = np.random.rand(num_nodes)
        cutoff_idx = 5

        # Create SpectralDecomposition
        sd = SpectralDecomposition(eigenvalues, eigenvectors, Dimension.GRAPH)

        # Filter signal
        filtered_signal = sd.filter_signal(signal, cutoff_idx)

        self.assertEqual(len(filtered_signal), num_nodes)

    def test_graph_laplacian_creation(self):
        # Create a simple adjacency matrix
        adjacency = np.array([[0, 1, 0],
                              [1, 0, 1],
                              [0, 1, 0]])

        # Create GraphLaplacian
        laplacian = GraphLaplacian.from_adjacency(adjacency)

        self.assertEqual(laplacian.matrix.shape, (3, 3))
        self.assertFalse(laplacian.is_normalized)

        # Check if the Laplacian matrix is correct
        expected_laplacian = np.array([[ 1, -1,  0],
                                      [-1,  2, -1],
                                      [ 0, -1,  1]])
        np.testing.assert_array_equal(laplacian.matrix, expected_laplacian)

    def test_graph_laplacian_creation_normalized(self):
        # Create a simple adjacency matrix
        adjacency = np.array([[0, 1, 0],
                              [1, 0, 1],
                              [0, 1, 0]])

        # Create normalized GraphLaplacian
        laplacian = GraphLaplacian.from_adjacency_normalized(adjacency)

        self.assertEqual(laplacian.matrix.shape, (3, 3))
        self.assertTrue(laplacian.is_normalized)

        # Check if the normalized Laplacian matrix is correct (approximate)
        expected_laplacian = np.array([[ 1.        , -0.70710678,  0.        ],
                                      [-0.70710678,  1.        , -0.70710678],
                                      [ 0.        , -0.70710678,  1.        ]])
        np.testing.assert_allclose(laplacian.matrix, expected_laplacian, atol=1e-7)

    def test_graph_laplacian_symmetric_check(self):
        # Create a non-symmetric matrix
        non_symmetric_matrix = np.array([[0, 1, 0],
                                         [0, 0, 1],
                                         [0, 1, 0]])

        # Expect ValueError
        with self.assertRaises(ValueError):
            GraphLaplacian(matrix=non_symmetric_matrix)

    def test_graph_laplacian_row_sums_check(self):
        # Create a matrix where row sums are not zero
        invalid_laplacian = np.array([[1, -1, 1],
                                      [-1, 2, -1],
                                      [0, -1, 1]])

        # Expect ValueError
        with self.assertRaises(ValueError):
            GraphLaplacian(matrix=invalid_laplacian)

    def test_graph_laplacian_spectral_decomposition(self):
        # Create a simple adjacency matrix
        adjacency = np.array([[0, 1, 0],
                              [1, 0, 1],
                              [0, 1, 0]])

        # Create GraphLaplacian
        laplacian = GraphLaplacian.from_adjacency(adjacency)

        # Perform spectral decomposition
        decomposition = laplacian.spectral_decomposition()

        self.assertIsInstance(decomposition, SpectralDecomposition)
        self.assertEqual(len(decomposition.eigenvalues), 3)
        self.assertEqual(decomposition.eigenvectors.shape, (3, 3))
        self.assertEqual(decomposition.dimension, Dimension.GRAPH)

    def test_graph_laplacian_connected_components(self):
        # Create a simple adjacency matrix for a connected graph
        adjacency_connected = np.array([[0, 1, 0],
                                        [1, 0, 1],
                                        [0, 1, 0]])

        # Create GraphLaplacian
        laplacian_connected = GraphLaplacian.from_adjacency(adjacency_connected)

        # Number of connected components should be 1
        self.assertEqual(laplacian_connected.connected_components, 1)

        # Create an adjacency matrix for a disconnected graph
        adjacency_disconnected = np.array([[0, 1, 0, 0],
                                           [1, 0, 0, 0],
                                           [0, 0, 0, 1],
                                           [0, 0, 1, 0]])

        # Create GraphLaplacian
        laplacian_disconnected = GraphLaplacian.from_adjacency(adjacency_disconnected)

        # Number of connected components should be 2
        self.assertEqual(laplacian_disconnected.connected_components, 2)

    def test_graph_laplacian_apply_gft(self):
        # Create a simple adjacency matrix
        adjacency = np.array([[0, 1, 0],
                              [1, 0, 1],
                              [0, 1, 0]])

        # Create GraphLaplacian
        laplacian = GraphLaplacian.from_adjacency(adjacency)

        # Create a signal
        signal = np.array([1, 2, 3])

        # Apply GFT
        fds = laplacian.apply_gft(signal)

        self.assertIsInstance(fds, FrequencyDomainSignal)
        self.assertEqual(len(fds.frequencies), 3)
        self.assertEqual(fds.dimension, Dimension.GRAPH)

    def test_graph_laplacian_filter_signal(self):
        # Create a simple adjacency matrix
        adjacency = np.array([[0, 1, 0],
                              [1, 0, 1],
                              [0, 1, 0]])

        # Create GraphLaplacian
        laplacian = GraphLaplacian.from_adjacency(adjacency)

        # Create a signal
        signal = np.array([1, 2, 3])
        cutoff_idx = 2

        # Filter signal
        aligned, liberal = laplacian.filter_signal(signal, cutoff_idx)

        self.assertEqual(len(aligned), 3)
        self.assertEqual(len(liberal), 3)

if __name__ == '__main__':
    unittest.main()