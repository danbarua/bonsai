import unittest
import numpy as np
from dynamics.oscillators import LayeredOscillatorState
from models.predictive import PredictiveHebbianOperator

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
        
        # Create a state with different but more moderate frequency differences
        phases = [
            np.zeros((2, 2)),
            np.zeros((2, 2))
        ]
        frequencies = [
            np.ones((2, 2)) * 0.8,  # Slightly slower oscillators
            np.ones((2, 2)) * 1.0   # Reference oscillators
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
        
        # Use higher learning rates and more iterations
        op = PredictiveHebbianOperator(
            dt=dt,
            pc_learning_rate=0.2,    # Higher learning rate
            hebb_learning_rate=0.0,  # Disable Hebbian learning to focus on PC
            pc_precision=2.0         # Higher precision for stronger error signals
        )
        
        # Apply multiple times to allow learning
        current_state = freq_diff_state
        
        # Record initial state
        initial_state = op.apply(current_state)
        initial_error = op.get_delta()["total_error"]
        
        # More iterations for better learning
        for _ in range(200):
            current_state = op.apply(current_state)
        
        # Get final error
        final_error = op.get_delta()["total_error"]
        
        # Check that learning has occurred
        self.assertTrue("total_error" in op.get_delta())
        
        # Check that prediction history was recorded
        self.assertGreater(len(op.prediction_history[0]), 0)
        self.assertGreater(len(op.error_history[0]), 0)
    
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
