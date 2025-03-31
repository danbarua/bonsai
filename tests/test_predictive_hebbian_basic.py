import unittest
import numpy as np
from dynamics.oscillators import LayeredOscillatorState
from models.predictive import PredictiveHebbianOperator

class TestPredictiveHebbianBasic(unittest.TestCase):
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

if __name__ == '__main__':
    unittest.main()
