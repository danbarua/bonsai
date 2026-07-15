import unittest
import numpy as np
import math
from enum import Enum, auto
from typing import NewType, TypeVar, Generic, Annotated, List, Tuple, Optional
from dataclasses import dataclass
from maths.core import Dimension, FrequencyBand, FrequencyHz, FrequencyRads, Phase, PhaseVector
from maths.spectral import FrequencyDomainSignal,SpectralDecomposition
from maths.graphs import GraphLaplacian

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