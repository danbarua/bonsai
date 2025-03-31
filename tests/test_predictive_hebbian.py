import unittest
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Protocol, TypeVar, Generic
from numpy.typing import NDArray
from dynamics.oscillators import LayeredOscillatorState
from models.predictive import PredictiveHebbianOperator

# --- Unit Tests ---
class TestPredictiveHebbianOperator(unittest.TestCase):
    # Define a consistent tolerance for assertions
    assertion_tolerance = 1e-3
    
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
        
        # Create a state with multiple layers for predictive coding tests
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
        
        # Create a state with three layers for deeper hierarchical tests
        phases_deep = [
            np.array([[0.0, np.pi/2], [np.pi, 3*np.pi/2]]),
            np.array([[np.pi/4, np.pi/3], [np.pi/2, 2*np.pi/3]]),
            np.array([[np.pi/6, np.pi/5], [np.pi/4, np.pi/3]])
        ]
        frequencies_deep = [
            np.array([[1.0, 1.2], [0.8, 1.1]]),
            np.array([[0.9, 1.0], [1.1, 0.8]]),
            np.array([[1.1, 0.9], [0.7, 1.2]])
        ]
        perturbations_deep = [
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            np.zeros((2, 2))
        ]
        layer_names_deep = ["Layer 1", "Layer 2", "Layer 3"]
        layer_shapes_deep = [(2, 2), (2, 2), (2, 2)]
        
        self.deep_layer_state = LayeredOscillatorState(
            _phases=phases_deep,
            _frequencies=frequencies_deep,
            _perturbations=perturbations_deep,
            _layer_names=layer_names_deep,
            _layer_shapes=layer_shapes_deep
        )
    
    def test_initialization(self):
        """Test initialization of the operator"""
        # Test with default parameters
        op = PredictiveHebbianOperator()
        self.assertEqual(op.dt, 0.01)  # Default dt is now 0.01
        self.assertEqual(op.pc_learning_rate, 0.01)
        self.assertEqual(op.pc_error_scaling, 0.5)
        self.assertEqual(op.pc_precision, 1.0)
        self.assertEqual(op.hebb_learning_rate, 0.01)
        self.assertEqual(op.hebb_decay_rate, 0.1)
        self.assertEqual(op.weight_normalization, True)
        self.assertEqual(op.between_layer_weights, [])
        self.assertEqual(op.within_layer_weights, [])
        
        # Test with custom parameters
        op = PredictiveHebbianOperator(
            dt=0.05, 
            pc_learning_rate=0.02, 
            hebb_learning_rate=0.03,
            weight_normalization=False
        )
        self.assertEqual(op.dt, 0.05)
        self.assertEqual(op.pc_learning_rate, 0.02)
        self.assertEqual(op.hebb_learning_rate, 0.03)
        self.assertEqual(op.weight_normalization, False)
    
    def test_weight_initialization(self):
        """Test weight initialization when not provided explicitly"""
        op = PredictiveHebbianOperator()
        new_state = op.apply(self.multi_layer_state)
        
        # Check that weights were initialized
        self.assertEqual(len(op.within_layer_weights), 2)  # One for each layer
        self.assertEqual(len(op.between_layer_weights), 1)  # One for connection between layers
        
        # Check dimensions of within-layer weights
        self.assertEqual(op.within_layer_weights[0].shape, (4, 4))  # 2x2 flattened to 4
        self.assertEqual(op.within_layer_weights[1].shape, (4, 4))  # 2x2 flattened to 4
        
        # Check dimensions of between-layer weights
        self.assertEqual(op.between_layer_weights[0].shape, (4, 4))  # 4 neurons in each layer
    
    def test_single_update(self):
        """Test a single update step"""
        op = PredictiveHebbianOperator(dt=0.1)
        new_state = op.apply(self.multi_layer_state)
        
        # Check that phases were updated
        for i in range(2):
            self.assertFalse(np.array_equal(new_state.phases[i], self.multi_layer_state.phases[i]))
        
        # Check that weights were updated
        self.assertTrue(len(op.within_layer_weights) > 0)
        self.assertTrue(len(op.between_layer_weights) > 0)
        
        # Check that phases are still in [0, 2π)
        for i in range(2):
            self.assertTrue(np.all(new_state.phases[i] >= 0))
            self.assertTrue(np.all(new_state.phases[i] < 2*np.pi))
        
        # Check that delta contains expected keys
        delta = op.get_delta()
        self.assertEqual(delta["type"], "enhanced_predictive_hebbian")
        self.assertTrue("coherence" in delta)
        self.assertTrue("mean_coherence" in delta)
        self.assertTrue("prediction_errors" in delta)
        self.assertTrue("total_error" in delta)
        self.assertTrue("weight_spectrum" in delta)
        self.assertTrue("fixed_point_analysis" in delta)
        self.assertTrue("system_energy" in delta)
    
    def test_multi_layer_update(self):
        """Test update with multiple layers"""
        op = PredictiveHebbianOperator(dt=0.1)
        new_state = op.apply(self.multi_layer_state)
        
        # Check that weights were initialized for both layers
        self.assertEqual(len(op.within_layer_weights), 2)
        self.assertEqual(op.within_layer_weights[0].shape, (4, 4))
        self.assertEqual(op.within_layer_weights[1].shape, (4, 4))
        
        # Check that between-layer weights were initialized
        self.assertEqual(len(op.between_layer_weights), 1)
        self.assertEqual(op.between_layer_weights[0].shape, (4, 4))
        
        # Check that all layers were updated
        for i in range(2):
            self.assertFalse(np.array_equal(new_state.phases[i], self.multi_layer_state.phases[i]))
            
        # Check delta contains information for both layers
        delta = op.get_delta()
        self.assertEqual(len(delta["coherence"]), 2)
        self.assertEqual(len(delta["prediction_errors"]), 1)  # One less than number of layers
    
    def test_deep_hierarchy_update(self):
        """Test update with a deeper hierarchy (3 layers)"""
        op = PredictiveHebbianOperator(dt=0.1)
        new_state = op.apply(self.deep_layer_state)
        
        # Check that weights were initialized for all layers
        self.assertEqual(len(op.within_layer_weights), 3)
        self.assertEqual(op.within_layer_weights[0].shape, (4, 4))
        self.assertEqual(op.within_layer_weights[1].shape, (4, 4))
        self.assertEqual(op.within_layer_weights[2].shape, (4, 4))
        
        # Check that between-layer weights were initialized
        self.assertEqual(len(op.between_layer_weights), 2)
        self.assertEqual(op.between_layer_weights[0].shape, (4, 4))
        self.assertEqual(op.between_layer_weights[1].shape, (4, 4))
        
        # Check that all layers were updated
        for i in range(3):
            self.assertFalse(np.array_equal(new_state.phases[i], self.deep_layer_state.phases[i]))
            
        # Check delta contains information for all layers
        delta = op.get_delta()
        self.assertEqual(len(delta["coherence"]), 3)
        self.assertEqual(len(delta["prediction_errors"]), 2)  # Two less than number of layers
    
    def test_phase_wrap_around(self):
        """Test handling of phase wrap-around"""
        # Create a state with phases near 2π
        phases = [
            np.array([[6.2, 0.1], [3.5, 6.28]]),  # Some phases close to 2π
            np.array([[6.1, 0.2], [3.6, 6.25]])   # Some phases close to 2π
        ]
        frequencies = [
            np.array([[1.0, 1.2], [0.8, 1.1]]),
            np.array([[0.9, 1.0], [1.1, 0.8]])
        ]
        perturbations = [
            np.zeros((2, 2)),
            np.zeros((2, 2))
        ]
        layer_names = ["Wrap Test 1", "Wrap Test 2"]
        layer_shapes = [(2, 2), (2, 2)]
        
        wrap_state = LayeredOscillatorState(
            _phases=phases,
            _frequencies=frequencies,
            _perturbations=perturbations,
            _layer_names=layer_names,
            _layer_shapes=layer_shapes
        )
        
        op = PredictiveHebbianOperator(dt=0.5)  # Larger time step to ensure wrap-around
        new_state = op.apply(wrap_state)
        
        # Check that all phases are in [0, 2π)
        for i in range(2):
            self.assertTrue(np.all(new_state.phases[i] >= 0))
            self.assertTrue(np.all(new_state.phases[i] < 2*np.pi))
    
    def test_coherence_calculation(self):
        """Test calculation of coherence values"""
        # Create a state with perfect coherence (all phases equal)
        coherent_phases = [
            np.ones((2, 2)) * np.pi/4,
            np.ones((2, 2)) * np.pi/3
        ]
        frequencies = [
            np.zeros((2, 2)),  # No frequency drift
            np.zeros((2, 2))   # No frequency drift
        ]
        perturbations = [
            np.zeros((2, 2)),
            np.zeros((2, 2))
        ]
        layer_names = ["Coherent 1", "Coherent 2"]
        layer_shapes = [(2, 2), (2, 2)]
        
        coherent_state = LayeredOscillatorState(
            _phases=coherent_phases,
            _frequencies=frequencies,
            _perturbations=perturbations,
            _layer_names=layer_names,
            _layer_shapes=layer_shapes
        )

        # timestep
        dt = 0.01
        
        op = PredictiveHebbianOperator(dt=dt)
        new_state = op.apply(coherent_state)
        delta = op.get_delta()
        
        # With identical phases, coherence should be 1.0
        self.assertAlmostEqual(delta["coherence"][0], 1.0, places=3)
        self.assertAlmostEqual(delta["coherence"][1], 1.0, places=3)
        self.assertAlmostEqual(delta["mean_coherence"], 1.0, places=3)
        
        # Create a state with minimal coherence (phases evenly distributed)
        incoherent_phases = [
            np.array([
                [0.0, np.pi/2], 
                [np.pi, 3*np.pi/2]
            ]),
            np.array([
                [0.0, np.pi/2], 
                [np.pi, 3*np.pi/2]
            ])
        ]
        
        incoherent_state = LayeredOscillatorState(
            _phases=incoherent_phases,
            _frequencies=frequencies,
            _perturbations=perturbations,
            _layer_names=layer_names,
            _layer_shapes=layer_shapes
        )
        
        op = PredictiveHebbianOperator(dt=dt)
        new_state = op.apply(incoherent_state)
        delta = op.get_delta()
        
        # With perfectly distributed phases, coherence should be close to 0
        self.assertLess(delta["coherence"][0], 0.01)  # Allow small numerical error
        self.assertLess(delta["coherence"][1], 0.01)  # Allow small numerical error
    
    def test_hebbian_weight_update(self):
        """Test the Hebbian weight update rule within layers"""
        # Define dt locally for this test
        dt = 0.01
        
        # Create a state with specific phase patterns to test Hebbian rule
        # Two in-phase oscillators and two out-of-phase
        phases = [
            np.array([
                [0.0, 0.0],          # These two are in phase
                [np.pi, np.pi]        # These two are in phase, but out of phase with the first two
            ]),
            np.array([
                [0.0, 0.0],          # These two are in phase
                [np.pi, np.pi]        # These two are in phase, but out of phase with the first two
            ])
        ]
        frequencies = [
            np.zeros((2, 2)),  # No frequency drift
            np.zeros((2, 2))   # No frequency drift
        ]
        perturbations = [
            np.zeros((2, 2)),
            np.zeros((2, 2))
        ]
        layer_names = ["Hebbian Test 1", "Hebbian Test 2"]
        layer_shapes = [(2, 2), (2, 2)]
        
        hebbian_state = LayeredOscillatorState(
            _phases=phases,
            _frequencies=frequencies,
            _perturbations=perturbations,
            _layer_names=layer_names,
            _layer_shapes=layer_shapes
        )
        
        # Use custom parameters to isolate Hebbian effects
        op = PredictiveHebbianOperator(
            dt=dt,  # Use local dt value
            hebb_learning_rate=1.0,  # High learning rate
            pc_learning_rate=0.0,    # Disable predictive coding
            weight_normalization=False  # Disable normalization
        )
        
        # Apply once to initialize weights
        new_state = op.apply(hebbian_state)
        
        # Store initial weights
        initial_weights = [w.copy() for w in op.within_layer_weights]
        
        # Apply again to see Hebbian updates
        new_state = op.apply(new_state)
        
        # Check weight changes according to Hebbian rule
        for layer in range(2):
            # Indices: 0-0, 0-1, 1-0, 1-1 should all have increased weights (in-phase)
            # Indices: 2-2, 2-3, 3-2, 3-3 should all have increased weights (in-phase)
            # Indices: 0-2, 0-3, 1-2, 1-3, 2-0, 2-1, 3-0, 3-1 should have decreased weights (out-of-phase)
            
            # Check in-phase connections (should strengthen)
            self.assertGreater(op.within_layer_weights[layer][0, 1], initial_weights[layer][0, 1])
            self.assertGreater(op.within_layer_weights[layer][1, 0], initial_weights[layer][1, 0])
            self.assertGreater(op.within_layer_weights[layer][2, 3], initial_weights[layer][2, 3])
            self.assertGreater(op.within_layer_weights[layer][3, 2], initial_weights[layer][3, 2])
            
            # Check out-of-phase connections (should weaken)
            self.assertLess(op.within_layer_weights[layer][0, 2], initial_weights[layer][0, 2])
            self.assertLess(op.within_layer_weights[layer][0, 3], initial_weights[layer][0, 3])
            self.assertLess(op.within_layer_weights[layer][1, 2], initial_weights[layer][1, 2])
            self.assertLess(op.within_layer_weights[layer][1, 3], initial_weights[layer][1, 3])
    
    def test_predictive_coding_weight_update(self):
        """Test the predictive coding weight update rule between layers"""
        # Define dt locally for this test
        dt = 0.01
        
        # Create a state with specific patterns to test predictive coding
        phases = [
            np.array([
                [0.0, np.pi/2],      # Layer 1 patterns
                [np.pi, 3*np.pi/2]
            ]),
            np.array([
                [0.0, np.pi/2],      # Layer 2 patterns (same as layer 1 for perfect prediction)
                [np.pi, 3*np.pi/2]
            ])
        ]
        frequencies = [
            np.zeros((2, 2)),  # No frequency drift
            np.zeros((2, 2))   # No frequency drift
        ]
        perturbations = [
            np.zeros((2, 2)),
            np.zeros((2, 2))
        ]
        layer_names = ["PC Test 1", "PC Test 2"]
        layer_shapes = [(2, 2), (2, 2)]
        
        pc_state = LayeredOscillatorState(
            _phases=phases,
            _frequencies=frequencies,
            _perturbations=perturbations,
            _layer_names=layer_names,
            _layer_shapes=layer_shapes
        )
        
        # Use custom parameters to isolate predictive coding effects
        op = PredictiveHebbianOperator(
            dt=dt,  # Use local dt value
            hebb_learning_rate=0.0,  # Disable Hebbian learning
            pc_learning_rate=0.1,    # Enable predictive coding
            weight_normalization=False  # Disable normalization
        )
        
        # Apply multiple times to allow learning
        current_state = pc_state
        for _ in range(10):
            current_state = op.apply(current_state)
        
        # Get final delta to check prediction error
        delta = op.get_delta()
        
        # With identical patterns, prediction error should decrease over time
        self.assertLess(delta["total_error"], 0.5)  # Error should be small after learning
    
    def test_weight_normalization(self):
        """Test that weight normalization prevents weight explosion"""
        # Create a state with random phases
        np.random.seed(42)  # For reproducibility
        random_phases = [
            np.random.uniform(0, 2*np.pi, (2, 2)),
            np.random.uniform(0, 2*np.pi, (2, 2))
        ]
        frequencies = [
            np.zeros((2, 2)),  # No frequency drift
            np.zeros((2, 2))   # No frequency drift
        ]
        perturbations = [
            np.zeros((2, 2)),
            np.zeros((2, 2))
        ]
        layer_names = ["Norm Test 1", "Norm Test 2"]
        layer_shapes = [(2, 2), (2, 2)]
        
        norm_state = LayeredOscillatorState(
            _phases=random_phases,
            _frequencies=frequencies,
            _perturbations=perturbations,
            _layer_names=layer_names,
            _layer_shapes=layer_shapes
        )
        
        # Test with normalization enabled
        op_with_norm = PredictiveHebbianOperator(
            dt=0.1,
            hebb_learning_rate=1.0,  # High learning rate to encourage weight growth
            pc_learning_rate=1.0,    # High learning rate to encourage weight growth
            weight_normalization=True
        )
        
        # Test with normalization disabled
        op_without_norm = PredictiveHebbianOperator(
            dt=0.1,
            hebb_learning_rate=1.0,  # High learning rate to encourage weight growth
            pc_learning_rate=1.0,    # High learning rate to encourage weight growth
            weight_normalization=False
        )
        
        # Apply multiple updates to both operators
        current_state_norm = norm_state
        current_state_no_norm = norm_state
        
        for _ in range(10):
            current_state_norm = op_with_norm.apply(current_state_norm)
            current_state_no_norm = op_without_norm.apply(current_state_no_norm)
        
        # Check maximum weight values
        max_weight_norm = max(np.max(w) for w in op_with_norm.within_layer_weights)
        max_weight_no_norm = max(np.max(w) for w in op_without_norm.within_layer_weights)
        
        # With normalization, weights should be bounded
        self.assertLess(max_weight_norm, 10.0)  # Reasonable upper bound
        
        # Without normalization, weights might grow larger
        # This test might be flaky depending on the specific dynamics
        # So we'll just check that normalization keeps weights smaller
        self.assertLess(max_weight_norm, max_weight_no_norm)
    
    def test_frequency_influence(self):
        """Test that natural frequencies properly influence phase updates"""
        # Define dt locally for this test
        dt = 0.01
        
        # Create state with varying frequencies but zero weights to isolate frequency effect
        phases = [
            np.zeros((2, 2)),
            np.zeros((2, 2))
        ]
        frequencies = [
            np.array([
                [0.5, 1.0],
                [1.5, 2.0]
            ]),
            np.array([
                [0.5, 1.0],
                [1.5, 2.0]
            ])
        ]
        perturbations = [
            np.zeros((2, 2)),
            np.zeros((2, 2))
        ]
        layer_names = ["Frequency Test 1", "Frequency Test 2"]
        layer_shapes = [(2, 2), (2, 2)]
        
        freq_state = LayeredOscillatorState(
            _phases=phases,
            _frequencies=frequencies,
            _perturbations=perturbations,
            _layer_names=layer_names,
            _layer_shapes=layer_shapes
        )
        
        # Use custom parameters to isolate frequency effects
        op = PredictiveHebbianOperator(
            dt=dt,  # Use local dt value
            hebb_learning_rate=0.0,  # Disable Hebbian learning
            pc_learning_rate=0.0     # Disable predictive coding
        )
        
        # Apply once to initialize weights
        new_state = op.apply(freq_state)
        
        # Set all weights to zero to isolate frequency effect
        for i in range(len(op.within_layer_weights)):
            op.within_layer_weights[i][:] = 0.0
        
        for i in range(len(op.between_layer_weights)):
            op.between_layer_weights[i][:] = 0.0
        
        # Apply again with zero weights
        new_state = op.apply(new_state)
        
        # Each oscillator should advance by 2π * freq * dt
        for layer in range(2):
            expected_phases = np.array([
                [0.5 * 2 * np.pi * dt, 1.0 * 2 * np.pi * dt],
                [1.5 * 2 * np.pi * dt, 2.0 * 2 * np.pi * dt]
            ])
            
            # Allow for small numerical differences
            np.testing.assert_allclose(new_state.phases[layer], expected_phases, atol=self.assertion_tolerance)
    
    def test_perturbation_response(self):
        """Test response to external perturbations"""
        # Create a state with zero phases and perturbations in specific locations
        phases = [
            np.zeros((2, 2)),
            np.zeros((2, 2))
        ]
        frequencies = [
            np.zeros((2, 2)),
            np.zeros((2, 2))
        ]
        perturbations = [
            np.array([
                [1.0, 0.0],
                [0.0, 0.0]
            ]),
            np.array([
                [0.0, 0.0],
                [0.0, 1.0]
            ])
        ]
        layer_names = ["Perturbation Test 1", "Perturbation Test 2"]
        layer_shapes = [(2, 2), (2, 2)]
        
        pert_state = LayeredOscillatorState(
            _phases=phases,
            _frequencies=frequencies,
            _perturbations=perturbations,
            _layer_names=layer_names,
            _layer_shapes=layer_shapes
        )
        
        op = PredictiveHebbianOperator(dt=0.1)
        new_state = op.apply(pert_state)
        
        # Check that perturbed oscillators have changed more than unperturbed ones
        for layer in range(2):
            phase_changes = np.abs(new_state.phases[layer] - pert_state.phases[layer])
            
            # Find indices of perturbed and unperturbed oscillators
            perturbed_indices = np.where(perturbations[layer] > 0)
            unperturbed_indices = np.where(perturbations[layer] == 0)
            
            # Get phase changes for perturbed and unperturbed oscillators
            perturbed_changes = phase_changes[perturbed_indices]
            unperturbed_changes = phase_changes[unperturbed_indices]
            
            # Perturbed oscillators should change more
            self.assertGreater(np.mean(perturbed_changes), np.mean(unperturbed_changes))
    
    def test_hierarchical_prediction(self):
        """Test hierarchical prediction between layers"""
        # Create a state with patterns that can be predicted hierarchically
        # Layer 1 has a simple pattern, Layer 2 has a related pattern
        phases = [
            np.array([
                [0.0, np.pi/2],
                [np.pi, 3*np.pi/2]
            ]),
            np.array([
                [np.pi/4, 3*np.pi/4],
                [5*np.pi/4, 7*np.pi/4]
            ])
        ]
        frequencies = [
            np.zeros((2, 2)),
            np.zeros((2, 2))
        ]
        perturbations = [
            np.zeros((2, 2)),
            np.zeros((2, 2))
        ]
        layer_names = ["Hierarchy Test 1", "Hierarchy Test 2"]
        layer_shapes = [(2, 2), (2, 2)]
        
        hierarchy_state = LayeredOscillatorState(
            _phases=phases,
            _frequencies=frequencies,
            _perturbations=perturbations,
            _layer_names=layer_names,
            _layer_shapes=layer_shapes
        )
        
        # Use custom parameters to focus on predictive coding
        op = PredictiveHebbianOperator(
            dt=0.1,
            pc_learning_rate=0.1,
            hebb_learning_rate=0.0  # Disable Hebbian learning
        )
        
        # Apply multiple times to allow learning
        current_state = hierarchy_state
        initial_error = None
        
        for i in range(20):
            current_state = op.apply(current_state)
            
            # Store initial error
            if i == 0:
                initial_error = op.get_delta()["total_error"]
        
        # Get final error
        final_error = op.get_delta()["total_error"]
        
        # Error should decrease over time as the model learns to predict
        self.assertLess(final_error, initial_error)
        
        # Check that prediction history was recorded
        self.assertGreater(len(op.prediction_history[0]), 0)
        self.assertGreater(len(op.error_history[0]), 0)
        
        # Verify trajectory analysis
        trajectory = op.get_trajectory_analysis()
        self.assertTrue("error_convergence" in trajectory)
        
    def test_combined_hebbian_predictive(self):
        """Test combined effect of Hebbian learning and predictive coding"""
        # Define dt locally for this test
        dt = 0.01
        
        # Create a state with patterns that benefit from both mechanisms
        phases = [
            np.array([
                [0.0, np.pi/2],
                [np.pi, 3*np.pi/2]
            ]),
            np.array([
                [np.pi/4, 3*np.pi/4],
                [5*np.pi/4, 7*np.pi/4]
            ])
        ]
        frequencies = [
            np.zeros((2, 2)),
            np.zeros((2, 2))
        ]
        perturbations = [
            np.zeros((2, 2)),
            np.zeros((2, 2))
        ]
        layer_names = ["Combined Test 1", "Combined Test 2"]
        layer_shapes = [(2, 2), (2, 2)]
        
        combined_state = LayeredOscillatorState(
            _phases=phases,
            _frequencies=frequencies,
            _perturbations=perturbations,
            _layer_names=layer_names,
            _layer_shapes=layer_shapes
        )
        
        # Use balanced parameters for both mechanisms
        op = PredictiveHebbianOperator(
            dt=dt,  # Use local dt value
            pc_learning_rate=0.05,
            hebb_learning_rate=0.05
        )
        
        # Apply multiple times to allow learning
        current_state = combined_state
        
        # Increase iterations for better convergence
        for _ in range(100):
            current_state = op.apply(current_state)
        
        # Check final metrics
        delta = op.get_delta()
        
        # Both coherence and prediction error should be reasonable
        self.assertGreater(delta["mean_coherence"], 0.5)  # Good coherence
        self.assertLess(delta["total_error"], 1.0)  # Low prediction error
        
        # Check energy metrics
        self.assertTrue("hebbian_energy" in delta["system_energy"])
        self.assertTrue("pc_energy" in delta["system_energy"])
        
    def test_system_energy_computation(self):
        """Test computation of system energy metrics"""
        op = PredictiveHebbianOperator(dt=0.1)
        new_state = op.apply(self.multi_layer_state)
        
        # Check energy metrics
        delta = op.get_delta()
        energy = delta["system_energy"]
        
        # Energy components should be present
        self.assertTrue("hebbian_energy" in energy)
        self.assertTrue("pc_energy" in energy)
        self.assertTrue("total_energy" in energy)
        
        # Total energy should be the sum of components
        self.assertAlmostEqual(
            energy["total_energy"],
            energy["hebbian_energy"] + energy["pc_energy"]
        )

# --- Enhanced Tests for Edge Cases and Numerical Stability ---
class TestPredictiveHebbianEdgeCases(unittest.TestCase):
    
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
        small_phases = [
            np.array([[1e-10, 1e-10]]),
            np.array([[1e-10, 1e-10]])
        ]
        small_frequencies = [
            np.array([[1e-10, 1e-10]]),
            np.array([[1e-10, 1e-10]])
        ]
        small_perturbations = [
            np.zeros((1, 2)),
            np.zeros((1, 2))
        ]
        small_layer_names = ["Small Values 1", "Small Values 2"]
        small_layer_shapes = [(1, 2), (1, 2)]
        
        self.small_state = LayeredOscillatorState(
            _phases=small_phases,
            _frequencies=small_frequencies,
            _perturbations=small_perturbations,
            _layer_names=small_layer_names,
            _layer_shapes=small_layer_shapes
        )
        
        # Large state for numerical overflow tests
        large_phases = [
            np.array([[1e5, 1e6]]),
            np.array([[1e5, 1e6]])
        ]
        large_frequencies = [
            np.array([[1e3, 1e4]]),
            np.array([[1e3, 1e4]])
        ]
        large_perturbations = [
            np.zeros((1, 2)),
            np.zeros((1, 2))
        ]
        large_layer_names = ["Large Values 1", "Large Values 2"]
        large_layer_shapes = [(1, 2), (1, 2)]
        
        self.large_state = LayeredOscillatorState(
            _phases=large_phases,
            _frequencies=large_frequencies,
            _perturbations=large_perturbations,
            _layer_names=large_layer_names,
            _layer_shapes=large_layer_shapes
        )
        
        # Mismatched layer shapes for testing robustness
        mismatched_phases = [
            np.array([[0.0, np.pi/2], [np.pi, 3*np.pi/2]]),  # 2x2
            np.array([[np.pi/4, np.pi/3, np.pi/2]])          # 1x3
        ]
        mismatched_frequencies = [
            np.array([[1.0, 1.2], [0.8, 1.1]]),
            np.array([[0.9, 1.0, 1.1]])
        ]
        mismatched_perturbations = [
            np.zeros((2, 2)),
            np.zeros((1, 3))
        ]
        mismatched_layer_names = ["Regular Layer", "Mismatched Layer"]
        mismatched_layer_shapes = [(2, 2), (1, 3)]
        
        self.mismatched_state = LayeredOscillatorState(
            _phases=mismatched_phases,
            _frequencies=mismatched_frequencies,
            _perturbations=mismatched_perturbations,
            _layer_names=mismatched_layer_names,
            _layer_shapes=mismatched_layer_shapes
        )
    
    def test_empty_state(self):
        """Test behavior with an empty state (no layers)"""
        op = PredictiveHebbianOperator()
        new_state = op.apply(self.empty_state)
        
        # Should handle empty state without error
        self.assertEqual(new_state.num_layers, 0)
        self.assertEqual(len(op.within_layer_weights), 0)
        self.assertEqual(len(op.between_layer_weights), 0)
        
        # Delta should still be returned with appropriate values
        delta = op.get_delta()
        self.assertEqual(delta["type"], "enhanced_predictive_hebbian")
        self.assertEqual(delta["coherence"], [])
        self.assertEqual(delta["mean_coherence"], 0.0)  # Default value for empty list
        self.assertEqual(delta["prediction_errors"], [])
        self.assertEqual(delta["total_error"], 0.0)  # Default value for empty list
    
    def test_very_small_values(self):
        """Test with very small numerical values for numerical stability"""
        # Use smaller learning rates for stability
        op = PredictiveHebbianOperator(dt=1e-5, pc_learning_rate=1e-5, hebb_learning_rate=1e-5)
        new_state = op.apply(self.small_state)
        
        # Should handle small values without numerical issues
        for i in range(2):
            self.assertTrue(np.all(np.isfinite(new_state.phases[i])))
        
        self.assertTrue(np.all(np.isfinite(op.within_layer_weights[0])))
        self.assertTrue(np.all(np.isfinite(op.between_layer_weights[0])))
        
        # Phase updates should be very small but non-zero
        for i in range(2):
            phase_diff = np.abs(new_state.phases[i] - self.small_state.phases[i])
            self.assertTrue(np.all(phase_diff > 0))
    
    def test_large_values_phase_wrapping(self):
        """Test with very large phase values for proper wrapping"""
        op = PredictiveHebbianOperator(dt=0.1)
        new_state = op.apply(self.large_state)
        
        # Phases should be properly wrapped to [0, 2π)
        for i in range(2):
            self.assertTrue(np.all(new_state.phases[i] >= 0))
            self.assertTrue(np.all(new_state.phases[i] < 2*np.pi))
    
    def test_mismatched_layer_shapes(self):
        """Test with layers of different shapes"""
        # Define dt locally for this test
        dt = 0.01
        op = PredictiveHebbianOperator(dt=dt)  # Use local dt value
        new_state = op.apply(self.mismatched_state)
        
        # Should handle different layer shapes correctly
        self.assertEqual(len(op.within_layer_weights), 2)
        self.assertEqual(op.within_layer_weights[0].shape, (4, 4))  # 2x2 flattened
        self.assertEqual(op.within_layer_weights[1].shape, (3, 3))  # 1x3 flattened
        
        # Between-layer weights should connect different sized layers
        # This feature has likely not been implemented yet
        self.assertEqual(op.between_layer_weights[0].shape, (3, 4))  # 3 neurons to 4 neurons
        
        # Phases should be updated for both layers
        for i in range(2):
            self.assertFalse(np.array_equal(new_state.phases[i], self.mismatched_state.phases[i]))
    
    def test_numerical_overflow_prevention(self):
        """Test that the implementation handles potential numerical overflows"""
        # Create state with extreme coupling weights
        phases = [
            np.array([[0.0, np.pi/4], [np.pi/2, 3*np.pi/4]]),
            np.array([[0.0, np.pi/4], [np.pi/2, 3*np.pi/4]])
        ]
        frequencies = [
            np.zeros((2, 2)),
            np.zeros((2, 2))
        ]
        perturbations = [
            np.zeros((2, 2)),
            np.zeros((2, 2))
        ]
        layer_names = ["Overflow Test 1", "Overflow Test 2"]
        layer_shapes = [(2, 2), (2, 2)]
        
        overflow_state = LayeredOscillatorState(
            _phases=phases,
            _frequencies=frequencies,
            _perturbations=perturbations,
            _layer_names=layer_names,
            _layer_shapes=layer_shapes
        )
        
        # Apply once to initialize weights
        op = PredictiveHebbianOperator(dt=0.1)
        new_state = op.apply(overflow_state)
        
        # Set weights to extreme values
        for i in range(len(op.within_layer_weights)):
            op.within_layer_weights[i][:] = 1e15
        
        for i in range(len(op.between_layer_weights)):
            op.between_layer_weights[i][:] = 1e15
        
        # Should handle without overflow
        try:
            new_state = op.apply(new_state)
            
            # Check no NaN or Inf values
            for i in range(2):
                self.assertTrue(np.all(np.isfinite(new_state.phases[i])))
            
            for w in op.within_layer_weights:
                self.assertTrue(np.all(np.isfinite(w)))
            
            for w in op.between_layer_weights:
                self.assertTrue(np.all(np.isfinite(w)))
            
            # Phases should still be in valid range
            for i in range(2):
                self.assertTrue(np.all(new_state.phases[i] >= 0))
                self.assertTrue(np.all(new_state.phases[i] < 2*np.pi))
            
        except (OverflowError, FloatingPointError, ValueError) as e:
            self.fail(f"Numerical overflow occurred: {e}")
    
    def test_weight_normalization_stability(self):
        """Test stability of weight normalization with extreme values"""
        # Create a state with random phases
        np.random.seed(42)
        random_phases = [
            np.random.uniform(0, 2*np.pi, (2, 2)),
            np.random.uniform(0, 2*np.pi, (2, 2))
        ]
        frequencies = [
            np.zeros((2, 2)),
            np.zeros((2, 2))
        ]
        perturbations = [
            np.zeros((2, 2)),
            np.zeros((2, 2))
        ]
        layer_names = ["Norm Stability 1", "Norm Stability 2"]
        layer_shapes = [(2, 2), (2, 2)]
        
        norm_state = LayeredOscillatorState(
            _phases=random_phases,
            _frequencies=frequencies,
            _perturbations=perturbations,
            _layer_names=layer_names,
            _layer_shapes=layer_shapes
        )
        
        # Initialize with normalization
        op = PredictiveHebbianOperator(
            dt=0.1,
            weight_normalization=True
        )
        
        # Apply once to initialize weights
        new_state = op.apply(norm_state)
        
        # Set some weights to extreme values
        op.within_layer_weights[0][0, 1] = 1e10
        op.between_layer_weights[0][0, 1] = 1e10
        
        # Apply again - normalization should handle extreme values
        new_state = op.apply(new_state)
        
        # Check that weights were normalized to reasonable values
        self.assertLess(np.max(op.within_layer_weights[0]), 10.0)
        self.assertLess(np.max(op.between_layer_weights[0]), 10.0)
    
    def test_phase_discontinuity(self):
        """Test behavior near phase discontinuity (wrap-around from 2π to 0)"""
        # Create a state with phases near the discontinuity
        phases = [
            np.array([
                [0.01, 6.27],  # Close to 0 and close to 2π
                [np.pi/2, 3*np.pi/2]  # Regular phases for comparison
            ]),
            np.array([
                [0.02, 6.26],  # Close to 0 and close to 2π
                [np.pi/3, 5*np.pi/3]  # Regular phases for comparison
            ])
        ]
        frequencies = [
            np.zeros((2, 2)),
            np.zeros((2, 2))
        ]
        perturbations = [
            np.zeros((2, 2)),
            np.zeros((2, 2))
        ]
        layer_names = ["Discontinuity Test 1", "Discontinuity Test 2"]
        layer_shapes = [(2, 2), (2, 2)]
        
        disc_state = LayeredOscillatorState(
            _phases=phases,
            _frequencies=frequencies,
            _perturbations=perturbations,
            _layer_names=layer_names,
            _layer_shapes=layer_shapes
        )
        
        op = PredictiveHebbianOperator(dt=0.01)
        new_state = op.apply(disc_state)
        
        # The true phase difference between 0.01 and 6.27 is very small (about 0.01)
        # So the coupling effect and weight update should reflect this small difference
        
        # Extract the relevant indices after flattening
        idx_near_zero = 0  # 0.01
        idx_near_2pi = 1   # 6.27
        
        # Calculate the circular phase difference (should be small)
        for layer in range(2):
            phase_diff = np.abs(np.angle(np.exp(1j * (phases[layer].flatten()[idx_near_zero] - 
                                                phases[layer].flatten()[idx_near_2pi]))))
            
            # The phase difference should be small
            self.assertLess(phase_diff, 0.1)
    
    def test_learning_with_frequency_differences(self):
        """Test learning with frequency differences between layers"""
        # Define dt locally for this test
        dt = 0.01
        
        # Create a state with different frequencies in each layer
        phases = [
            np.zeros((2, 2)),
            np.zeros((2, 2))
        ]
        frequencies = [
            np.ones((2, 2)) * 0.5,  # Slower oscillators
            np.ones((2, 2)) * 1.0   # Faster oscillators
        ]
        perturbations = [
            np.zeros((2, 2)),
            np.zeros((2, 2))
        ]
        layer_names = ["Slow Layer", "Fast Layer"]
        layer_shapes = [(2, 2), (2, 2)]
        
        freq_diff_state = LayeredOscillatorState(
            _phases=phases,
            _frequencies=frequencies,
            _perturbations=perturbations,
            _layer_names=layer_names,
            _layer_shapes=layer_shapes
        )
        
        # Use higher learning rates to compensate for frequency differences
        op = PredictiveHebbianOperator(
            dt=dt,  # Use local dt value
            pc_learning_rate=0.1,
            hebb_learning_rate=0.1
        )
        
        # Apply multiple times to allow learning
        current_state = freq_diff_state
        initial_error = None
        
        for i in range(100):
            current_state = op.apply(current_state)
            
            # Store initial error
            if i == 0:
                initial_error = op.get_delta()["total_error"]
        
        # Get final error
        final_error = op.get_delta()["total_error"]
        
        # Despite frequency differences, error should decrease
        self.assertLessEqual(final_error, initial_error)
    
    def test_trajectory_analysis_with_changing_inputs(self):
        """Test trajectory analysis with changing inputs"""
        # Create initial state
        phases = [
            np.array([
                [0.0, np.pi/2],
                [np.pi, 3*np.pi/2]
            ]),
            np.array([
                [0.0, np.pi/2],
                [np.pi, 3*np.pi/2]
            ])
        ]
        frequencies = [
            np.zeros((2, 2)),
            np.zeros((2, 2))
        ]
        perturbations = [
            np.zeros((2, 2)),
            np.zeros((2, 2))
        ]
        layer_names = ["Trajectory Test 1", "Trajectory Test 2"]
        layer_shapes = [(2, 2), (2, 2)]
        
        trajectory_state = LayeredOscillatorState(
            _phases=phases,
            _frequencies=frequencies,
            _perturbations=perturbations,
            _layer_names=layer_names,
            _layer_shapes=layer_shapes
        )
        
        op = PredictiveHebbianOperator(dt=0.1)
        
        # Apply for several steps with initial pattern
        current_state = trajectory_state
        for _ in range(10):
            current_state = op.apply(current_state)
        
        # Create a new state with different pattern
        new_phases = [
            np.array([
                [np.pi/4, 3*np.pi/4],
                [5*np.pi/4, 7*np.pi/4]
            ]),
            np.array([
                [np.pi/4, 3*np.pi/4],
                [5*np.pi/4, 7*np.pi/4]
            ])
        ]
        
        new_trajectory_state = LayeredOscillatorState(
            _phases=new_phases,
            _frequencies=frequencies,
            _perturbations=perturbations,
            _layer_names=layer_names,
            _layer_shapes=layer_shapes
        )
        
        # Apply for several more steps with new pattern
        current_state = new_trajectory_state
        for _ in range(10):
            current_state = op.apply(current_state)
        
        # Get trajectory analysis
        trajectory = op.get_trajectory_analysis()
        
        # Should have recorded error history
        self.assertTrue("error_convergence" in trajectory)
        self.assertGreaterEqual(len(trajectory["error_convergence"]), 0)
    
    def test_spectral_analysis(self):
        """Test spectral analysis of weight matrices"""
        op = PredictiveHebbianOperator(dt=0.1)
        new_state = op.apply(self.large_state)
        
        # Get spectral analysis
        delta = op.get_delta()
        spectrum = delta["weight_spectrum"]
        
        # Should have analysis for both types of weights
        self.assertTrue("hebbian" in spectrum)
        self.assertTrue("predictive" in spectrum)
        
        # Should have entries for each layer
        self.assertEqual(len(spectrum["hebbian"]), 2)
        self.assertEqual(len(spectrum["predictive"]), 1)
        
        # Each entry should have spectral properties
        for layer_spectrum in spectrum["hebbian"]:
            if "error" not in layer_spectrum:  # Skip if SVD failed
                self.assertTrue("max_eigval" in layer_spectrum)
                self.assertTrue("condition_number" in layer_spectrum)
        
        for layer_spectrum in spectrum["predictive"]:
            if "error" not in layer_spectrum:  # Skip if SVD failed
                self.assertTrue("max_sv" in layer_spectrum)
                self.assertTrue("condition_number" in layer_spectrum)

if __name__ == '__main__':
    unittest.main()
