import unittest
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Protocol, TypeVar, Generic
from numpy.typing import NDArray

# --- Mock LayeredOscillatorState for testing ---
@dataclass
class LayeredOscillatorState:
    """Test version of LayeredOscillatorState"""
    _phases: list[NDArray[np.float64]]
    _frequencies: list[NDArray[np.float64]]
    _perturbations: list[NDArray[np.float64]]
    _layer_names: list[str]
    _layer_shapes: list[tuple[int, ...]]
    
    @property
    def phases(self) -> list[NDArray[np.float64]]:
        return self._phases
    
    @property
    def frequencies(self) -> list[NDArray[np.float64]]:
        return self._frequencies
        
    @property
    def perturbations(self) -> list[NDArray[np.float64]]:
        return self._perturbations
    
    @property
    def layer_names(self) -> list[str]:
        return self._layer_names
    
    @property
    def layer_shapes(self) -> list[tuple[int, ...]]:
        return self._layer_shapes
    
    @property
    def num_layers(self) -> int:
        return len(self._phases)
    
    def copy(self) -> 'LayeredOscillatorState':
        return LayeredOscillatorState(
            _phases=[phase.copy() for phase in self._phases],
            _frequencies=[freq.copy() for freq in self._frequencies],
            _perturbations=[pert.copy() for pert in self._perturbations],
            _layer_names=self._layer_names.copy(),
            _layer_shapes=self._layer_shapes.copy()
        )

# --- Protocol and Implementation Classes ---
S = TypeVar('S')

class StateMutation(Protocol, Generic[S]):
    def apply(self, state: S) -> S: ...
    def get_delta(self) -> dict[str, Any]: ...

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

        # Ensure weight diagonal is zero'd
        for i in range(layer_count):
            np.fill_diagonal(self.weights[i], 0.0)  # Explicit zero diagonal
        
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
            phase_update_flat += state.frequencies[i].flatten() * 2 * np.pi
            
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
        
        for i in range(layer_count):
            # Phase coherence
            z = np.exp(1j * new_state.phases[i].flatten())
            coherence = float(np.abs(np.mean(z)))
            coherence_values.append(coherence)
            
            # Weight statistics
            mean_weight = float(np.mean(self.weights[i]))
            mean_weights.append(mean_weight)
        
        # Store information about this update
        self.last_delta = {
            "type": "hebbian_kuramoto",
            "coherence": coherence_values,
            "mean_coherence": float(np.mean(coherence_values)),
            "mean_weights": mean_weights,
            "max_weight": float(np.max([np.max(w) for w in self.weights]))
        }
        
        return new_state
    
    def get_delta(self) -> dict[str, Any]:
        return self.last_delta

# --- Unit Tests ---
class TestHebbianKuramotoOperator(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a simple state with one layer of 2x2 oscillators
        phases = [np.array([[0.0, np.pi/2], [np.pi, 3*np.pi/2]])]
        frequencies = [np.array([[1.0, 1.2], [0.8, 1.1]])]
        perturbations = [np.zeros((2, 2))]
        layer_names = ["Test Layer"]
        layer_shapes = [(2, 2)]
        
        self.state = LayeredOscillatorState(
            _phases=phases,
            _frequencies=frequencies,
            _perturbations=perturbations,
            _layer_names=layer_names,
            _layer_shapes=layer_shapes
        )
        
        # Create a state with multiple layers
        phases_multi = [
            np.array([[0.0, np.pi/2], [np.pi, 3*np.pi/2]]),
            np.array([[np.pi/4, np.pi/3], [np.pi/2, 2*np.pi/3]])
        ]
        frequencies_multi = [
            np.array([[1.0, 1.2], [0.8, 1.1]]),
            np.array([[0.9, 1.0], [1.1, 0.8]])
        ]
        perturbations_multi = [
            np.zeros((2, 2)),
            np.zeros((2, 2))
        ]
        layer_names_multi = ["Layer 1", "Layer 2"]
        layer_shapes_multi = [(2, 2), (2, 2)]
        
        self.multi_layer_state = LayeredOscillatorState(
            _phases=phases_multi,
            _frequencies=frequencies_multi,
            _perturbations=perturbations_multi,
            _layer_names=layer_names_multi,
            _layer_shapes=layer_shapes_multi
        )
        
        # Predefined weights for testing
        self.test_weights = [np.ones((4, 4)) * 0.5]
        
    def test_initialization(self):
        """Test initialization of the operator"""
        # Test with default parameters
        op = HebbianKuramotoOperator()
        self.assertEqual(op.dt, 0.1)
        self.assertEqual(op.mu, 0.01)
        self.assertEqual(op.alpha, 0.1)
        self.assertEqual(op.weights, [])
        
        # Test with custom parameters
        op = HebbianKuramotoOperator(dt=0.05, mu=0.02, alpha=0.2)
        self.assertEqual(op.dt, 0.05)
        self.assertEqual(op.mu, 0.02)
        self.assertEqual(op.alpha, 0.2)
        
        # Test with initial weights
        op = HebbianKuramotoOperator(init_weights=self.test_weights)
        self.assertEqual(len(op.weights), 1)
        np.testing.assert_array_equal(op.weights[0], self.test_weights[0])
    
    def test_weight_initialization(self):
        """Test weight initialization when not provided explicitly"""
        op = HebbianKuramotoOperator()
        new_state = op.apply(self.state)
        
        # Check that weights were initialized
        self.assertEqual(len(op.weights), 1)
        self.assertEqual(op.weights[0].shape, (4, 4))  # 2x2 flattened to 4
    
    def test_single_update(self):
        """Test a single update step"""
        op = HebbianKuramotoOperator(init_weights=self.test_weights, dt=0.1, mu=0.1, alpha=0.1)
        new_state = op.apply(self.state)
        
        # Check that phases were updated
        self.assertFalse(np.array_equal(new_state.phases[0], self.state.phases[0]))
        
        # Check that weights were updated
        self.assertFalse(np.array_equal(op.weights[0], self.test_weights[0]))
        
        # Check that phases are still in [0, 2π)
        self.assertTrue(np.all(new_state.phases[0] >= 0))
        self.assertTrue(np.all(new_state.phases[0] < 2*np.pi))
        
        # Check that delta contains expected keys
        delta = op.get_delta()
        self.assertEqual(delta["type"], "hebbian_kuramoto")
        self.assertTrue("coherence" in delta)
        self.assertTrue("mean_coherence" in delta)
        self.assertTrue("mean_weights" in delta)
        self.assertTrue("max_weight" in delta)
    
    def test_multi_layer_update(self):
        """Test update with multiple layers"""
        op = HebbianKuramotoOperator(dt=0.1, mu=0.1, alpha=0.1)
        new_state = op.apply(self.multi_layer_state)
        
        # Check that weights were initialized for both layers
        self.assertEqual(len(op.weights), 2)
        self.assertEqual(op.weights[0].shape, (4, 4))
        self.assertEqual(op.weights[1].shape, (4, 4))
        
        # Check that all layers were updated
        for i in range(2):
            self.assertFalse(np.array_equal(new_state.phases[i], self.multi_layer_state.phases[i]))
            
        # Check delta contains information for both layers
        delta = op.get_delta()
        self.assertEqual(len(delta["coherence"]), 2)
        self.assertEqual(len(delta["mean_weights"]), 2)
    
    def test_phase_wrap_around(self):
        """Test handling of phase wrap-around"""
        # Create a state with phases near 2π
        phases = [np.array([[6.2, 0.1], [3.5, 6.28]])]  # Some phases close to 2π
        frequencies = [np.array([[1.0, 1.2], [0.8, 1.1]])]
        perturbations = [np.zeros((2, 2))]
        layer_names = ["Wrap Test"]
        layer_shapes = [(2, 2)]
        
        wrap_state = LayeredOscillatorState(
            _phases=phases,
            _frequencies=frequencies,
            _perturbations=perturbations,
            _layer_names=layer_names,
            _layer_shapes=layer_shapes
        )
        
        op = HebbianKuramotoOperator(dt=0.5)  # Larger time step to ensure wrap-around
        new_state = op.apply(wrap_state)
        
        # Check that all phases are in [0, 2π)
        self.assertTrue(np.all(new_state.phases[0] >= 0))
        self.assertTrue(np.all(new_state.phases[0] < 2*np.pi))
    
    def test_coherence_calculation(self):
        """Test calculation of coherence values"""
        # Create a state with perfect coherence (all phases equal)
        coherent_phases = [np.ones((2, 2)) * np.pi/4]
        frequencies = [np.zeros((2, 2))]  # No frequency drift
        perturbations = [np.zeros((2, 2))]
        layer_names = ["Coherent"]
        layer_shapes = [(2, 2)]
        
        coherent_state = LayeredOscillatorState(
            _phases=coherent_phases,
            _frequencies=frequencies,
            _perturbations=perturbations,
            _layer_names=layer_names,
            _layer_shapes=layer_shapes
        )
        
        op = HebbianKuramotoOperator(dt=0.1)
        new_state = op.apply(coherent_state)
        delta = op.get_delta()
        
        # With identical phases, coherence should be 1.0
        self.assertAlmostEqual(delta["coherence"][0], 1.0, places=5)
        self.assertAlmostEqual(delta["mean_coherence"], 1.0, places=5)
        
        # Create a state with minimal coherence (phases evenly distributed)
        incoherent_phases = [np.array([
            [0.0, np.pi/2], 
            [np.pi, 3*np.pi/2]
        ])]
        
        incoherent_state = LayeredOscillatorState(
            _phases=incoherent_phases,
            _frequencies=frequencies,
            _perturbations=perturbations,
            _layer_names=layer_names,
            _layer_shapes=layer_shapes
        )
        
        op = HebbianKuramotoOperator(dt=0.1)
        new_state = op.apply(incoherent_state)
        delta = op.get_delta()
        
        # With perfectly distributed phases, coherence should be close to 0
        self.assertLess(delta["coherence"][0], 0.01)  # Allow small numerical error
    
    def test_hebbian_weight_update(self):
        """Test the Hebbian weight update rule"""
        # Create a state with specific phase patterns to test Hebbian rule
        # Two in-phase oscillators and two out-of-phase
        phases = [np.array([
            [0.0, 0.0],          # These two are in phase
            [np.pi, np.pi]        # These two are in phase, but out of phase with the first two
        ])]
        frequencies = [np.zeros((2, 2))]  # No frequency drift
        perturbations = [np.zeros((2, 2))]
        layer_names = ["Hebbian Test"]
        layer_shapes = [(2, 2)]
        
        hebbian_state = LayeredOscillatorState(
            _phases=phases,
            _frequencies=frequencies,
            _perturbations=perturbations,
            _layer_names=layer_names,
            _layer_shapes=layer_shapes
        )
        
        # Start with zero weights
        zero_weights = [np.zeros((4, 4))]
        
        op = HebbianKuramotoOperator(init_weights=zero_weights, dt=0.1, mu=1.0, alpha=0.0)
        new_state = op.apply(hebbian_state)

        for k, v in op.last_delta.items():
            print(f"DEBUG: {k} = {v}")
        
        # Check that weights between in-phase oscillators increased
        # Indices: 0-0, 0-1, 1-0, 1-1 should all have cos(0) = 1
        # Indices: 2-2, 2-3, 3-2, 3-3 should all have cos(0) = 1
        self.assertAlmostEqual(op.weights[0][0, 0], 0.0)  # Diagonal doesn't change
        self.assertAlmostEqual(op.weights[0][0, 1], 0.1)  # mu * dt * cos(0)
        self.assertAlmostEqual(op.weights[0][2, 2], 0.0)  # Diagonal doesn't change
        self.assertAlmostEqual(op.weights[0][2, 3], 0.1)  # mu * dt * cos(0)
        
        # Weights between out-of-phase oscillators should decrease
        # Indices: 0-2, 0-3, 1-2, 1-3 should all have cos(π) = -1
        self.assertAlmostEqual(op.weights[0][0, 2], -0.1)  # mu * dt * cos(π)
        self.assertAlmostEqual(op.weights[0][1, 3], -0.1)  # mu * dt * cos(π)
    
    def test_weight_decay(self):
        """Test that weights decay properly"""
        # Create uniform weights
        uniform_weights = [np.ones((4, 4))]
        
        # Create a state with all zeros to isolate decay effect
        phases = [np.zeros((2, 2))]
        frequencies = [np.zeros((2, 2))]
        perturbations = [np.zeros((2, 2))]
        layer_names = ["Decay Test"]
        layer_shapes = [(2, 2)]
        
        decay_state = LayeredOscillatorState(
            _phases=phases,
            _frequencies=frequencies,
            _perturbations=perturbations,
            _layer_names=layer_names,
            _layer_shapes=layer_shapes
        )
        
        # With zero phases, cos(θi-θj) = 1.0, so we'll have 
        # mu * 1.0 - alpha * w as update
        op = HebbianKuramotoOperator(init_weights=uniform_weights, dt=1.0, mu=0.0, alpha=0.1)
        new_state = op.apply(decay_state)
        
        # All weights should decay by alpha * dt
        for i in range(4):
            for j in range(4):
                self.assertAlmostEqual(op.weights[0][i, j], 0.9)  # 1.0 - alpha * dt
    
    def test_frequency_influence(self):
        """Test that natural frequencies properly influence phase updates"""
        # Create state with varying frequencies but zero weights to isolate frequency effect
        phases = [np.zeros((2, 2))]
        frequencies = [np.array([
            [0.5, 1.0],
            [1.5, 2.0]
        ])]
        perturbations = [np.zeros((2, 2))]
        layer_names = ["Frequency Test"]
        layer_shapes = [(2, 2)]
        
        freq_state = LayeredOscillatorState(
            _phases=phases,
            _frequencies=frequencies,
            _perturbations=perturbations,
            _layer_names=layer_names,
            _layer_shapes=layer_shapes
        )
        
        # Use zero weights to isolate frequency effect
        zero_weights = [np.zeros((4, 4))]
        
        op = HebbianKuramotoOperator(init_weights=zero_weights, dt=0.1)
        new_state = op.apply(freq_state)
        
        # Each oscillator should advance by 2π * freq * dt
        expected_phases = np.array([
            [0.5 * 2 * np.pi * 0.1, 1.0 * 2 * np.pi * 0.1],
            [1.5 * 2 * np.pi * 0.1, 2.0 * 2 * np.pi * 0.1]
        ])
        
        np.testing.assert_allclose(new_state.phases[0], expected_phases)
    
    def test_stability_at_fixed_point(self):
        """Test stability of the system near the theoretical fixed point"""
        # Create a state with uniform phases
        phases = [np.ones((2, 2)) * np.pi/4]
        frequencies = [np.zeros((2, 2))]  # No frequency drift
        perturbations = [np.zeros((2, 2))]
        layer_names = ["Fixed Point Test"]
        layer_shapes = [(2, 2)]
        
        fp_state = LayeredOscillatorState(
            _phases=phases,
            _frequencies=frequencies,
            _perturbations=perturbations,
            _layer_names=layer_names,
            _layer_shapes=layer_shapes
        )
        
        # Set weights to theoretical fixed point: γ_ij = cos(θ_i - θ_j)/α
        phases_flat = phases[0].flatten()
        phase_diffs = phases_flat[:, np.newaxis] - phases_flat[np.newaxis, :]
        alpha = 0.1
        fp_weights = [np.cos(phase_diffs) / alpha]
        
        op = HebbianKuramotoOperator(init_weights=fp_weights, dt=0.1, mu=0.1, alpha=alpha)
        
        # Apply multiple updates and check stability
        current_state = fp_state
        initial_weights = op.weights[0].copy()
        
        for _ in range(10):
            current_state = op.apply(current_state)
        
        # Phases should remain stable (not changing much)
        np.testing.assert_allclose(current_state.phases[0], phases[0], atol=1e-5)
        
        # Weights should remain stable (not changing much)
        np.testing.assert_allclose(op.weights[0], initial_weights, atol=1e-5)
    
    def test_coupling_dynamics(self):
        """Test the coupling effect on phase dynamics"""
        # Create a state with two types of oscillators (fast and slow)
        phases = [np.array([
            [0.0, 0.0],  # Oscillators starting in-phase
            [np.pi, np.pi]  # Oscillators starting in-phase but out of phase with first group
        ])]
        frequencies = [np.array([
            [1.0, 1.0],  # Same frequency group
            [2.0, 2.0]   # Same frequency group
        ])]
        perturbations = [np.zeros((2, 2))]
        layer_names = ["Coupling Test"]
        layer_shapes = [(2, 2)]
        
        coupling_state = LayeredOscillatorState(
            _phases=phases,
            _frequencies=frequencies,
            _perturbations=perturbations,
            _layer_names=layer_names,
            _layer_shapes=layer_shapes
        )
        
        # Set up symmetric positive coupling between all oscillators
        coupling_weights = [np.ones((4, 4)) * 0.5]
        coupling_weights[0] = coupling_weights[0] - np.diag(np.diag(coupling_weights[0]))  # Zero diagonal
        
        op = HebbianKuramotoOperator(init_weights=coupling_weights, dt=0.1, mu=0.0, alpha=0.0)
        
        # Run for several steps to observe coupling effect
        current_state = coupling_state
        
        for _ in range(10):
            current_state = op.apply(current_state)
        
        # With positive coupling, oscillators of the same frequency should synchronize
        # Check if phase differences within each group are smaller than between groups
        phases_flat = current_state.phases[0].flatten()
        
        # Phase difference between oscillators 0 and 1 (same frequency)
        diff_0_1 = np.abs(np.angle(np.exp(1j * (phases_flat[0] - phases_flat[1]))))
        
        # Phase difference between oscillators 0 and 2 (different frequencies)
        diff_0_2 = np.abs(np.angle(np.exp(1j * (phases_flat[0] - phases_flat[2]))))
        
        self.assertLess(diff_0_1, diff_0_2)

# --- Enhanced Tests for Edge Cases and Numerical Stability ---
class TestHebbianKuramotoEdgeCases(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures for edge cases"""
        # Empty state for empty layer tests
        self.empty_state = LayeredOscillatorState(
            _phases=[],
            _frequencies=[],
            _perturbations=[],
            _layer_names=[],
            _layer_shapes=[]
        )
        
        # Very small state for numerical precision tests
        small_phases = [np.array([[1e-10, 1e-10]])]
        small_frequencies = [np.array([[1e-10, 1e-10]])]
        small_perturbations = [np.zeros((1, 2))]
        small_layer_names = ["Small Values"]
        small_layer_shapes = [(1, 2)]
        
        self.small_state = LayeredOscillatorState(
            _phases=small_phases,
            _frequencies=small_frequencies,
            _perturbations=small_perturbations,
            _layer_names=small_layer_names,
            _layer_shapes=small_layer_shapes
        )
        
        # Large state for numerical overflow tests
        large_phases = [np.array([[1e5, 1e6]])]
        large_frequencies = [np.array([[1e3, 1e4]])]
        large_perturbations = [np.zeros((1, 2))]
        large_layer_names = ["Large Values"]
        large_layer_shapes = [(1, 2)]
        
        self.large_state = LayeredOscillatorState(
            _phases=large_phases,
            _frequencies=large_frequencies,
            _perturbations=large_perturbations,
            _layer_names=large_layer_names,
            _layer_shapes=large_layer_shapes
        )
    
    def test_empty_state(self):
        """Test behavior with an empty state (no layers)"""
        op = HebbianKuramotoOperator()
        new_state = op.apply(self.empty_state)
        
        # Should handle empty state without error
        self.assertEqual(new_state.num_layers, 0)
        self.assertEqual(len(op.weights), 0)
        
        # Delta should still be returned with appropriate values
        delta = op.get_delta()
        self.assertEqual(delta["type"], "hebbian_kuramoto")
        self.assertEqual(delta["coherence"], [])
        self.assertEqual(delta["mean_coherence"], 0.0)  # Default value for empty list
        self.assertEqual(delta["mean_weights"], [])
        self.assertEqual(delta["max_weight"], 0.0)  # Default value for empty list
    
    def test_very_small_values(self):
        """Test with very small numerical values for numerical stability"""
        # Use smaller learning rates for stability
        op = HebbianKuramotoOperator(dt=1e-5, mu=1e-5, alpha=1e-5)
        new_state = op.apply(self.small_state)
        
        # Should handle small values without numerical issues
        self.assertTrue(np.all(np.isfinite(new_state.phases[0])))
        self.assertTrue(np.all(np.isfinite(op.weights[0])))
        
        # Phase updates should be very small but non-zero
        phase_diff = np.abs(new_state.phases[0] - self.small_state.phases[0])
        self.assertTrue(np.all(phase_diff > 0))
    
    def test_large_values_phase_wrapping(self):
        """Test with very large phase values for proper wrapping"""
        op = HebbianKuramotoOperator(dt=0.1)
        new_state = op.apply(self.large_state)
        
        # Phases should be properly wrapped to [0, 2π)
        self.assertTrue(np.all(new_state.phases[0] >= 0))
        self.assertTrue(np.all(new_state.phases[0] < 2*np.pi))
    
    def test_phase_difference_symmetry(self):
        """Test that phase differences are handled with proper symmetry"""
        # Create oscillators with specific phase differences
        phases = [np.array([
            [0.0, np.pi/2],
            [np.pi, 3*np.pi/2]
        ])]
        frequencies = [np.zeros((2, 2))]
        perturbations = [np.zeros((2, 2))]
        layer_names = ["Symmetry Test"]
        layer_shapes = [(2, 2)]
        
        symmetry_state = LayeredOscillatorState(
            _phases=phases,
            _frequencies=frequencies,
            _perturbations=perturbations,
            _layer_names=layer_names,
            _layer_shapes=layer_shapes
        )
        
        # Custom weights to test symmetry
        sym_weights = [np.ones((4, 4))]
        
        op = HebbianKuramotoOperator(init_weights=sym_weights, dt=0.1, mu=0.1, alpha=0.1)
        new_state = op.apply(symmetry_state)
        
        # For the Hebbian rule with cos(θj-θi), W[i,j] should update according to cos(θj-θi)
        # and W[j,i] should update according to cos(θi-θj) = cos(-(θj-θi)) = cos(θj-θi)
        # So the weight matrix should remain symmetric
        
        # Check weight matrix symmetry
        for i in range(4):
            for j in range(i+1, 4):
                self.assertAlmostEqual(op.weights[0][i, j], op.weights[0][j, i], 
                                      msg=f"Weights not symmetric for {i},{j} and {j},{i}")
    
    # def test_phase_discontinuity(self):
    #     """Test behavior near phase discontinuity (wrap-around from 2π to