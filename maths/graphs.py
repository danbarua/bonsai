# === Graph Laplacian Types ===
import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Annotated, Generic, List, NewType, Optional, Tuple, TypeVar

import numpy as np
from numpy.typing import NDArray


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