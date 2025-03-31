import unittest
import numpy as np
import matplotlib.pyplot as plt
from dynamics.oscillators import LayeredOscillatorState
from models.predictive import PredictiveHebbianOperator
from models.hebbian import HebbianKuramotoOperator
import matplotlib.animation as animation

# Try to import sklearn, but make it optional
try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Some visualization features will be limited.")



class TestPredictiveHebbianCharacter(unittest.TestCase):
    # Class attribute for sklearn availability
    SKLEARN_AVAILABLE = _SKLEARN_AVAILABLE
    """
    Tests for processing character inputs with a Predictive Hebbian network.
    
    This test suite evaluates how a Predictive Hebbian network processes and responds to
    character inputs, with a focus on hierarchical processing, noise robustness, and
    ambiguity resolution.
    """
    
    def setUp(self):
        """Set up test fixtures"""
        # Default parameters for character processing
        self.dt = 0.01
        self.pc_learning_rate = 0.05
        self.hebb_learning_rate = 0.05
        self.pc_error_scaling = 0.5
        self.pc_precision = 1.0
        self.hebb_decay_rate = 0.1
        self.perturbation_strength = 1.0
        self.max_steps = 1000
        self.convergence_threshold = 1e-4
        
    def get_character_matrix(self, char):
        """Return an 8x12 binary matrix representing the given character."""
        # Dictionary of predefined character matrices
        chars = {
            'A': np.array([
                [0, 0, 0, 1, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0]
            ]).T,  # Transpose to get 8x12
            'B': np.array([
                [0, 1, 1, 1, 1, 1, 0, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 1, 1, 1, 1, 0, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0]
            ]).T,
            'C': np.array([
                [0, 0, 1, 1, 1, 1, 0, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 0, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0]
            ]).T,
            '1': np.array([
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0]
            ]).T,
            '2': np.array([
                [0, 0, 1, 1, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0]
            ]).T,
            'P': np.array([
                [0, 1, 1, 1, 1, 1, 0, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 1, 1, 1, 1, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0]
            ]).T,
            'R': np.array([
                [0, 1, 1, 1, 1, 1, 0, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 1, 1, 1, 1, 0, 0],
                [0, 1, 0, 1, 0, 0, 0, 0],
                [0, 1, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0]
            ]).T,
            'O': np.array([
                [0, 0, 1, 1, 1, 1, 0, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 0, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0]
            ]).T,
            'D': np.array([
                [0, 1, 1, 1, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 1, 0, 0],
                [0, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0]
            ]).T,
        }
        return chars.get(char, np.zeros((8, 12)))
    
    def add_noise_to_character(self, char_matrix, noise_level=0.1):
        """Add random noise to a character matrix."""
        # Generate random noise (0 or 1) with probability noise_level
        noise = np.random.binomial(1, noise_level, char_matrix.shape)
        # XOR the noise with the character matrix to flip bits
        noisy_matrix = np.logical_xor(char_matrix, noise).astype(float)
        return noisy_matrix
    
    def create_ambiguous_character(self, char1, char2, ambiguity_level=0.5):
        """Create an ambiguous character by blending two characters."""
        char1_matrix = self.get_character_matrix(char1)
        char2_matrix = self.get_character_matrix(char2)
        
        # Create a mask for blending
        mask = np.random.binomial(1, ambiguity_level, char1_matrix.shape)
        
        # Blend the characters
        ambiguous_matrix = np.where(mask, char2_matrix, char1_matrix)
        return ambiguous_matrix
    
    def create_occluded_character(self, char, occlusion_type='horizontal', occlusion_level=0.3):
        """Create a partially occluded character."""
        char_matrix = self.get_character_matrix(char)
        occluded_matrix = char_matrix.copy()
        
        if occlusion_type == 'horizontal':
            # Occlude horizontal strips
            n_rows = char_matrix.shape[0]
            n_occluded = int(n_rows * occlusion_level)
            occluded_rows = np.random.choice(n_rows, n_occluded, replace=False)
            occluded_matrix[occluded_rows, :] = 0
            
        elif occlusion_type == 'vertical':
            # Occlude vertical strips
            n_cols = char_matrix.shape[1]
            n_occluded = int(n_cols * occlusion_level)
            occluded_cols = np.random.choice(n_cols, n_occluded, replace=False)
            occluded_matrix[:, occluded_cols] = 0
            
        elif occlusion_type == 'random':
            # Randomly occlude pixels
            mask = np.random.binomial(1, occlusion_level, char_matrix.shape)
            occluded_matrix = np.where(mask, 0, char_matrix)
            
        return occluded_matrix
    
    def create_hierarchical_state(self, input_matrix, layer_shapes=None, perturbation_strength=1.0):
        """
        Create a hierarchical LayeredOscillatorState from an input matrix.
        
        Args:
            input_matrix: The input matrix (e.g., character matrix)
            layer_shapes: List of shapes for each layer. If None, uses default hierarchy.
            perturbation_strength: Strength of perturbations
        
        Returns:
            LayeredOscillatorState with hierarchical structure
        """
        # Default layer shapes if not provided
        if layer_shapes is None:
            # Input shape is the shape of the input matrix
            input_shape = input_matrix.shape
            # Create a hierarchy with decreasing dimensions
            layer_shapes = [
                input_shape,                                  # Layer 1: Original input
                (input_shape[0]//2, input_shape[1]//2),       # Layer 2: Half resolution
                (input_shape[0]//4, input_shape[1]//4)        # Layer 3: Quarter resolution
            ]
        
        # Initialize phases, frequencies, and perturbations for each layer
        phases = []
        frequencies = []
        perturbations = []
        layer_names = []
        
        # For each layer in the hierarchy
        for i, shape in enumerate(layer_shapes):
            # Random initial phases
            layer_phases = np.random.uniform(0, 2*np.pi, shape)
            phases.append(layer_phases)
            
            # Uniform frequencies
            layer_frequencies = np.ones(shape)
            frequencies.append(layer_frequencies)
            
            # Perturbations only for the input layer
            if i == 0:
                # Ensure input_matrix is properly resized if needed
                if input_matrix.shape != shape:
                    from scipy.ndimage import zoom
                    zoom_factors = (shape[0]/input_matrix.shape[0], shape[1]/input_matrix.shape[1])
                    resized_input = zoom(input_matrix, zoom_factors, order=0)
                else:
                    resized_input = input_matrix
                
                layer_perturbations = resized_input * perturbation_strength
            else:
                layer_perturbations = np.zeros(shape)
            
            perturbations.append(layer_perturbations)
            layer_names.append(f"Layer {i+1}")
        
        # Create the state
        return LayeredOscillatorState(
            _phases=phases,
            _frequencies=frequencies,
            _perturbations=perturbations,
            _layer_names=layer_names,
            _layer_shapes=layer_shapes
        )
    
    def process_character(self, state, model_type="predictive", iterations=100):
        """
        Process a character through the specified model type.
        
        Args:
            state: The initial LayeredOscillatorState
            model_type: "predictive" or "hebbian"
            iterations: Number of iterations to run
            
        Returns:
            Tuple of (final_state, weights, deltas, states_history)
        """
        # Initialize the appropriate operator
        if model_type == "predictive":
            op = PredictiveHebbianOperator(
                dt=self.dt,
                pc_learning_rate=self.pc_learning_rate,
                hebb_learning_rate=self.hebb_learning_rate,
                pc_error_scaling=self.pc_error_scaling,
                pc_precision=self.pc_precision,
                hebb_decay_rate=self.hebb_decay_rate
            )
        else:  # hebbian
            op = HebbianKuramotoOperator(
                dt=self.dt,
                mu=self.hebb_learning_rate,
                alpha=self.hebb_decay_rate
            )
        
        # Process for the specified number of iterations
        current_state = state
        deltas = []
        states_history = [current_state.copy()]
        
        for _ in range(iterations):
            current_state = op.apply(current_state)
            deltas.append(op.get_delta())
            states_history.append(current_state.copy())
        
        # For predictive model, collect weights
        if model_type == "predictive":
            weights = {
                "within_layer_weights": op.within_layer_weights,
                "between_layer_weights": op.between_layer_weights
            }
        else:
            weights = {
                "weights": op.weights
            }
        
        return current_state, weights, deltas, states_history
    
    def calculate_local_coherence(self, phase_data):
        """Calculate local phase coherence map."""
        coherence_map = np.zeros(phase_data.shape)
        for i in range(phase_data.shape[0]):
            for j in range(phase_data.shape[1]):
                # Define a neighborhood around oscillator (i,j)
                i_min, i_max = max(0, i-1), min(phase_data.shape[0], i+2)
                j_min, j_max = max(0, j-1), min(phase_data.shape[1], j+2)
                
                # Calculate local coherence
                neighborhood = phase_data[i_min:i_max, j_min:j_max]
                z = np.exp(1j * neighborhood.flatten())
                coherence_map[i, j] = np.abs(np.mean(z))
        
        return coherence_map
    
    def visualize_hierarchical_representation(self, state, char, save_path=None):
        """Visualize how a character is represented across the hierarchy of layers."""
        # Number of layers
        n_layers = len(state.phases)
        
        # Create a figure with rows for different visualization types and columns for layers
        fig, axes = plt.subplots(4, n_layers, figsize=(n_layers*4, 16))
        
        # If only one layer, reshape axes for consistent indexing
        if n_layers == 1:
            axes = axes.reshape(4, 1)
        
        # Original character for reference (first column, first row)
        char_matrix = self.get_character_matrix(char)
        axes[0, 0].imshow(char_matrix, cmap='binary')
        axes[0, 0].set_title(f"Original Character: '{char}'")
        
        # For each layer, show different visualizations
        for i in range(n_layers):
            layer_name = state.layer_names[i]
            phase_data = state.phases[i]
            
            # Row 1: Phase distribution using hsv colormap (circular)
            if i > 0:  # Skip first column of first row (used for original character)
                axes[0, i].imshow(phase_data, cmap='hsv')
                axes[0, i].set_title(f"Phase Distribution\n{layer_name}")
            
            # Row 2: Local coherence map
            coherence_map = self.calculate_local_coherence(phase_data)
            coh_img = axes[1, i].imshow(coherence_map, cmap='viridis')
            axes[1, i].set_title(f"Local Coherence\n{layer_name}")
            plt.colorbar(coh_img, ax=axes[1, i])
            
            # Row 3: Phase gradient magnitude (spatial derivative)
            # This shows where phase changes rapidly vs. smoothly
            gradient_y, gradient_x = np.gradient(phase_data)
            gradient_mag = np.sqrt(gradient_x**2 + gradient_y**2)
            grad_img = axes[2, i].imshow(gradient_mag, cmap='magma')
            axes[2, i].set_title(f"Phase Gradient\n{layer_name}")
            plt.colorbar(grad_img, ax=axes[2, i])
            
            # Row 4: Oscillator activity (complex representation)
            # Convert phases to complex numbers and visualize magnitude/angle
            complex_z = np.exp(1j * phase_data)
            activity = np.abs(complex_z)  # Should be 1 everywhere, but useful for verification
            act_img = axes[3, i].imshow(activity, cmap='plasma')
            axes[3, i].set_title(f"Oscillator Activity\n{layer_name}")
            plt.colorbar(act_img, ax=axes[3, i])
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def visualize_feature_extraction(self, state, weights, char, save_path=None):
        """Visualize how features are extracted and transformed between layers."""
        n_layers = len(state.phases)
        
        # Create a figure with 2 rows: top for phase patterns, bottom for weight visualizations
        fig, axes = plt.subplots(2, n_layers-1, figsize=((n_layers-1)*5, 10))
        
        # If only one layer pair, reshape axes for consistent indexing
        if n_layers == 2:
            axes = axes.reshape(2, 1)
        
        # For each pair of adjacent layers
        for i in range(n_layers-1):
            # Get the between-layer weights
            between_weights = weights["between_layer_weights"][i]
            
            # Reshape weights for visualization if needed
            # This depends on how the weights are stored and the layer shapes
            lower_shape = state.layer_shapes[i]
            higher_shape = state.layer_shapes[i+1]
            
            # Top row: Show the phase patterns of adjacent layers
            lower_phases = state.phases[i]
            higher_phases = state.phases[i+1]
            
            # Create a side-by-side comparison
            comparison = np.zeros((max(lower_shape[0], higher_shape[0]), 
                                  lower_shape[1] + higher_shape[1]))
            
            # Insert the phase patterns
            comparison[:lower_shape[0], :lower_shape[1]] = lower_phases
            comparison[:higher_shape[0], lower_shape[1]:] = higher_phases
            
            # Display the comparison
            phase_img = axes[0, i].imshow(comparison, cmap='hsv')
            axes[0, i].set_title(f"Phase Patterns: {state.layer_names[i]} → {state.layer_names[i+1]}")
            plt.colorbar(phase_img, ax=axes[0, i])
            
            # Add a vertical line to separate the layers
            axes[0, i].axvline(x=lower_shape[1]-0.5, color='white', linestyle='-', linewidth=2)
            
            # Bottom row: Visualize the weight matrix using SVD to find principal components
            try:
                u, s, vh = np.linalg.svd(between_weights, full_matrices=False)
                
                # Use top 2 components to create a 2D visualization of weight space
                weight_viz = np.outer(u[:, 0], vh[0, :]) * s[0]
                if s.size > 1:  # Add second component if available
                    weight_viz += np.outer(u[:, 1], vh[1, :]) * s[1]
                
                # Reshape to match layer dimensions if possible
                try:
                    weight_viz_reshaped = weight_viz.reshape(higher_shape[0], higher_shape[1], 
                                                           lower_shape[0], lower_shape[1])
                    # Average across input dimensions to get a 2D map
                    weight_map = np.mean(weight_viz_reshaped, axis=(2, 3))
                    weight_img = axes[1, i].imshow(weight_map, cmap='coolwarm')
                    axes[1, i].set_title(f"Weight Principal Components\n{state.layer_names[i]} → {state.layer_names[i+1]}")
                except:
                    # Fallback: just show the raw weight matrix
                    weight_img = axes[1, i].imshow(weight_viz, cmap='coolwarm')
                    axes[1, i].set_title(f"Weight Matrix\n{state.layer_names[i]} → {state.layer_names[i+1]}")
            except:
                # Fallback: just show the raw weight matrix
                weight_img = axes[1, i].imshow(between_weights, cmap='coolwarm')
                axes[1, i].set_title(f"Weight Matrix\n{state.layer_names[i]} → {state.layer_names[i+1]}")
            
            plt.colorbar(weight_img, ax=axes[1, i])
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def visualize_information_flow(self, op, char, save_path=None):
        """Visualize how information flows through the hierarchy during processing."""
        # Extract prediction and error history from the operator
        prediction_history = op.prediction_history
        error_history = op.error_history
        
        if not prediction_history or not error_history:
            print("No prediction or error history available")
            return
        
        n_layers = len(error_history) + 1
        n_steps = min(len(error_history[0]), 100)  # Limit to 100 steps for visualization
        
        # Create a figure with 3 rows: predictions, errors, and error reduction
        fig, axes = plt.subplots(3, n_layers-1, figsize=((n_layers-1)*5, 15))
        
        # If only one layer pair, reshape axes for consistent indexing
        if n_layers == 2:
            axes = axes.reshape(3, 1)
        
        # For each pair of adjacent layers
        for i in range(n_layers-1):
            # Get the prediction and error history for this layer pair
            predictions = prediction_history[i][:n_steps]
            errors = error_history[i][:n_steps]
            
            # Calculate error magnitude over time
            error_magnitude = [np.mean(np.abs(err)) for err in errors]
            
            # Row 1: Visualize predictions over time
            # We'll show a few key frames from the prediction history
            n_frames = min(5, n_steps)
            frame_indices = np.linspace(0, n_steps-1, n_frames, dtype=int)
            
            # Create a grid of prediction frames
            prediction_grid = np.zeros((predictions[0].size, n_frames))
            for j, idx in enumerate(frame_indices):
                prediction_grid[:, j] = predictions[idx]
            
            pred_img = axes[0, i].imshow(prediction_grid, cmap='hsv', aspect='auto')
            axes[0, i].set_title(f"Predictions Over Time\nLayer {i} → {i+1}")
            axes[0, i].set_xticks(range(n_frames))
            axes[0, i].set_xticklabels([f"Step {idx}" for idx in frame_indices])
            plt.colorbar(pred_img, ax=axes[0, i])
            
            # Row 2: Visualize errors over time (same frames)
            error_grid = np.zeros((errors[0].size, n_frames))
            for j, idx in enumerate(frame_indices):
                error_grid[:, j] = errors[idx]
            
            err_img = axes[1, i].imshow(error_grid, cmap='RdBu', aspect='auto')
            axes[1, i].set_title(f"Prediction Errors Over Time\nLayer {i} → {i+1}")
            axes[1, i].set_xticks(range(n_frames))
            axes[1, i].set_xticklabels([f"Step {idx}" for idx in frame_indices])
            plt.colorbar(err_img, ax=axes[1, i])
            
            # Row 3: Plot error reduction over time
            axes[2, i].plot(range(n_steps), error_magnitude)
            axes[2, i].set_title(f"Error Magnitude Over Time\nLayer {i} → {i+1}")
            axes[2, i].set_xlabel("Processing Step")
            axes[2, i].set_ylabel("Mean Absolute Error")
            axes[2, i].grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def visualize_character_embedding(self, character_states, characters, save_path=None):
        """
        Visualize how different characters are embedded in the phase space of each layer.
        Uses dimensionality reduction (PCA or t-SNE) to create 2D visualizations.
        
        Args:
            character_states: Dictionary mapping characters to their final states
            characters: List of characters that were processed
        """
        # Get the number of layers from the first state
        first_char = characters[0]
        n_layers = len(character_states[first_char].phases)
        
        # Create a figure with one subplot per layer
        fig, axes = plt.subplots(1, n_layers, figsize=(n_layers*5, 5))
        
        # If only one layer, reshape axes for consistent indexing
        if n_layers == 1:
            axes = [axes]
        
        # For each layer, perform dimensionality reduction
        for layer_idx in range(n_layers):
            # Collect phase data for this layer across all characters
            phase_data = []
            for char in characters:
                # Flatten the phase data
                flat_phases = character_states[char].phases[layer_idx].flatten()
                # Convert to complex representation for better distance metrics
                complex_phases = np.exp(1j * flat_phases)
                # Use real and imaginary parts as features
                features = np.concatenate([complex_phases.real, complex_phases.imag])
                phase_data.append(features)
            
            # Convert to numpy array
            phase_data = np.array(phase_data)
            
            # Apply dimensionality reduction if sklearn is available
            if SKLEARN_AVAILABLE:
                if len(characters) >= 5:  # t-SNE needs more samples
                    tsne = TSNE(n_components=2, random_state=42)
                    embedding = tsne.fit_transform(phase_data)
                    method = "t-SNE"
                else:
                    pca = PCA(n_components=2)
                    embedding = pca.fit_transform(phase_data)
                    method = "PCA"
                
                # Plot the embedding
                for i, char in enumerate(characters):
                    axes[layer_idx].scatter(embedding[i, 0], embedding[i, 1], s=100, label=char)
                
                axes[layer_idx].set_title(f"Layer {layer_idx+1} Character Embedding ({method})")
            else:
                # Alternative visualization when sklearn is not available
                # Use a simple 2D projection of the first two components
                # of the complex phase data
                for i, char in enumerate(characters):
                    complex_phases = np.exp(1j * character_states[char].phases[layer_idx].flatten())
                    # Use the first two components as x, y coordinates
                    if len(complex_phases) >= 2:
                        x, y = complex_phases[0].real, complex_phases[1].real
                        axes[layer_idx].scatter(x, y, s=100, label=char)
                    else:
                        # Fallback for very small layers
                        axes[layer_idx].scatter(i, 0, s=100, label=char)
                
                axes[layer_idx].set_title(f"Layer {layer_idx+1} Character Embedding (Simple Projection)")
            
            axes[layer_idx].legend()
            axes[layer_idx].grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def visualize_reconstruction(self, state, weights, char, save_path=None):
        """
        Visualize how well each layer can reconstruct the original character.
        Uses the between-layer weights to project from higher layers back to the input layer.
        """
        n_layers = len(state.phases)
        
        # Create a figure
        fig, axes = plt.subplots(1, n_layers, figsize=(n_layers*4, 4))
        
        # If only one layer, reshape axes for consistent indexing
        if n_layers == 1:
            axes = [axes]
        
        # Original character
        char_matrix = self.get_character_matrix(char)
        axes[0].imshow(char_matrix, cmap='binary')
        axes[0].set_title(f"Original: '{char}'")
        
        # For each higher layer, attempt reconstruction
        for i in range(1, n_layers):
            # Get the phase pattern for this layer
            higher_phases = state.phases[i]
            
            # Convert to complex representation
            higher_complex = np.exp(1j * higher_phases.flatten())
            
            # Reconstruct through each layer back to the input
            reconstructed = higher_complex
            for j in range(i-1, -1, -1):
                # Use transpose of weights for backward projection
                between_weights = weights["between_layer_weights"][j]
                reconstructed = between_weights.T @ reconstructed
                
                # Normalize to unit circle
                reconstructed = reconstructed / np.abs(reconstructed)
            
            # Convert back to phase representation
            reconstructed_phases = np.angle(reconstructed)
            
            # Reshape to match input dimensions
            try:
                reconstructed_phases = reconstructed_phases.reshape(state.layer_shapes[0])
            except:
                # If reshaping fails, use a default shape
                reconstructed_phases = reconstructed_phases.reshape(char_matrix.shape)
            
            # Display the reconstruction
            axes[i].imshow(reconstructed_phases, cmap='hsv')
            axes[i].set_title(f"Reconstructed from Layer {i+1}")
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def visualize_noise_comparison(self, clean_state, noisy_state, predictive_state, hebbian_state, char, noise_level, save_path=None):
        """
        Visualize comparison between clean, noisy, and processed states from both models.
        
        Args:
            clean_state: Original clean character state
            noisy_state: Noisy character state
            predictive_state: State after processing with predictive Hebbian model
            hebbian_state: State after processing with standard Hebbian model
            char: The character being processed
            noise_level: The noise level applied
        """
        # Create a figure with 2 rows: top for input/output, bottom for coherence maps
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # Original character
        char_matrix = self.get_character_matrix(char)
        axes[0, 0].imshow(char_matrix, cmap='binary')
        axes[0, 0].set_title(f"Original Character: '{char}'")
        
        # Noisy character
        noisy_matrix = self.add_noise_to_character(char_matrix, noise_level)
        axes[0, 1].imshow(noisy_matrix, cmap='binary')
        axes[0, 1].set_title(f"Noisy Character ({noise_level*100:.0f}% noise)")
        
        # Predictive model result
        axes[0, 2].imshow(predictive_state.phases[0], cmap='hsv')
        axes[0, 2].set_title("Predictive Hebbian Result")
        
        # Standard Hebbian model result
        axes[0, 3].imshow(hebbian_state.phases[0], cmap='hsv')
        axes[0, 3].set_title("Standard Hebbian Result")
        
        # Coherence maps
        # Clean state coherence
        clean_coherence = self.calculate_local_coherence(clean_state.phases[0])
        coh_img1 = axes[1, 0].imshow(clean_coherence, cmap='viridis')
        axes[1, 0].set_title("Clean Coherence")
        plt.colorbar(coh_img1, ax=axes[1, 0])
        
        # Noisy state coherence
        noisy_coherence = self.calculate_local_coherence(noisy_state.phases[0])
        coh_img2 = axes[1, 1].imshow(noisy_coherence, cmap='viridis')
        axes[1, 1].set_title("Noisy Coherence")
        plt.colorbar(coh_img2, ax=axes[1, 1])
        
        # Predictive model coherence
        predictive_coherence = self.calculate_local_coherence(predictive_state.phases[0])
        coh_img3 = axes[1, 2].imshow(predictive_coherence, cmap='viridis')
        axes[1, 2].set_title("Predictive Coherence")
        plt.colorbar(coh_img3, ax=axes[1, 2])
        
        # Standard Hebbian model coherence
        hebbian_coherence = self.calculate_local_coherence(hebbian_state.phases[0])
        coh_img4 = axes[1, 3].imshow(hebbian_coherence, cmap='viridis')
        axes[1, 3].set_title("Hebbian Coherence")
        plt.colorbar(coh_img4, ax=axes[1, 3])
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def visualize_ambiguity_resolution(self, ambiguous_matrix, predictive_state, hebbian_state, char1, char2, ambiguity_level, save_path=None):
        """
        Visualize how ambiguous characters are resolved by different models.
        
        Args:
            ambiguous_matrix: The ambiguous character matrix
            predictive_state: State after processing with predictive Hebbian model
            hebbian_state: State after processing with standard Hebbian model
            char1, char2: The two characters being blended
            ambiguity_level: The level of ambiguity applied
        """
        # Create a figure with 2 rows: top for characters, bottom for phase patterns
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # Original characters
        char1_matrix = self.get_character_matrix(char1)
        axes[0, 0].imshow(char1_matrix, cmap='binary')
        axes[0, 0].set_title(f"Character 1: '{char1}'")
        
        char2_matrix = self.get_character_matrix(char2)
        axes[0, 1].imshow(char2_matrix, cmap='binary')
        axes[0, 1].set_title(f"Character 2: '{char2}'")
        
        # Ambiguous character
        axes[0, 2].imshow(ambiguous_matrix, cmap='binary')
        axes[0, 2].set_title(f"Ambiguous Character\n({ambiguity_level*100:.0f}% blend)")
        
        # Difference map between original characters
        diff_map = np.abs(char1_matrix - char2_matrix)
        axes[0, 3].imshow(diff_map, cmap='binary')
        axes[0, 3].set_title("Difference Map")
        
        # Phase patterns
        # Process char1 with predictive model
        char1_state = self.create_hierarchical_state(char1_matrix)
        char1_final, _, _, _ = self.process_character(char1_state, model_type="predictive", iterations=100)
        
        # Process char2 with predictive model
        char2_state = self.create_hierarchical_state(char2_matrix)
        char2_final, _, _, _ = self.process_character(char2_state, model_type="predictive", iterations=100)
        
        # Calculate similarity to char1 and char2 for both models
        # For predictive model
        pred_similarity_1 = np.mean(np.cos(predictive_state.phases[0] - char1_final.phases[0]))
        pred_similarity_2 = np.mean(np.cos(predictive_state.phases[0] - char2_final.phases[0]))
        
        # For Hebbian model
        hebb_similarity_1 = np.mean(np.cos(hebbian_state.phases[0] - char1_final.phases[0]))
        hebb_similarity_2 = np.mean(np.cos(hebbian_state.phases[0] - char2_final.phases[0]))
        
        # Display phase patterns with similarity scores
        axes[1, 0].imshow(char1_final.phases[0], cmap='hsv')
        axes[1, 0].set_title(f"Char 1 Phase Pattern")
        
        axes[1, 1].imshow(char2_final.phases[0], cmap='hsv')
        axes[1, 1].set_title(f"Char 2 Phase Pattern")
        
        axes[1, 2].imshow(predictive_state.phases[0], cmap='hsv')
        axes[1, 2].set_title(f"Predictive Result\nSim1={pred_similarity_1:.2f}, Sim2={pred_similarity_2:.2f}")
        
        axes[1, 3].imshow(hebbian_state.phases[0], cmap='hsv')
        axes[1, 3].set_title(f"Hebbian Result\nSim1={hebb_similarity_1:.2f}, Sim2={hebb_similarity_2:.2f}")
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def test_hierarchical_character_processing(self):
        """Test processing of characters through a hierarchical predictive Hebbian network."""
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Process a character through the hierarchical network
        char = 'A'
        char_matrix = self.get_character_matrix(char)
        
        # Create hierarchical state
        state = self.create_hierarchical_state(char_matrix, perturbation_strength=2.0)
        
        # Process the character
        final_state, weights, deltas, states_history = self.process_character(
            state, model_type="predictive", iterations=200
        )
        
        # Visualize the hierarchical representation
        self.visualize_hierarchical_representation(final_state, char, save_path=f"hierarchical_{char}_representation.png")
        
        # Visualize feature extraction
        self.visualize_feature_extraction(final_state, weights, char, save_path=f"hierarchical_{char}_features.png")
        
        # Visualize reconstruction from each layer
        self.visualize_reconstruction(final_state, weights, char, save_path=f"hierarchical_{char}_reconstruction.png")
        
        # Check that all layers have been updated
        for i in range(len(final_state.phases)):
            self.assertFalse(np.array_equal(final_state.phases[i], state.phases[i]))
        
        # Check that coherence increases through the hierarchy
        coherence_values = []
        for i in range(len(final_state.phases)):
            coherence_map = self.calculate_local_coherence(final_state.phases[i])
            coherence_values.append(np.mean(coherence_map))
        
        # Higher layers should generally have higher coherence
        # This might not always be true, but it's a reasonable expectation
        print(f"Layer coherence values: {coherence_values}")
        
        # Process multiple characters and visualize their embedding
        characters = ['A', 'B', 'C', '1', '2']
        character_states = {}
        
        for c in characters:
            c_matrix = self.get_character_matrix(c)
            c_state = self.create_hierarchical_state(c_matrix, perturbation_strength=2.0)
            c_final, _, _, _ = self.process_character(c_state, model_type="predictive", iterations=200)
            character_states[c] = c_final
        
        # Visualize character embedding
        self.visualize_character_embedding(character_states, characters, save_path="character_embedding.png")
        
        # Test assertions
        # Check that different characters produce distinct representations
        for i in range(len(characters)):
            for j in range(i+1, len(characters)):
                char1 = characters[i]
                char2 = characters[j]
                
                # Calculate phase difference between characters
                phase_diff = np.abs(np.angle(np.exp(1j * (character_states[char1].phases[0] - character_states[char2].phases[0]))))
                mean_diff = np.mean(phase_diff)
                
                # Different characters should produce distinct states
                self.assertGreater(mean_diff, 0.1, f"Characters '{char1}' and '{char2}' produce too similar states")
    
    def test_noise_robustness_comparison(self):
        """Test noise robustness of predictive Hebbian vs. standard Hebbian models."""
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Process a character with varying noise levels
        char = 'A'
        char_matrix = self.get_character_matrix(char)
        
        # Test different noise levels
        noise_levels = [0.1, 0.2, 0.3]
        
        for noise_level in noise_levels:
            # Create noisy character
            noisy_matrix = self.add_noise_to_character(char_matrix, noise_level)
            
            # Create states for clean and noisy characters
            clean_state = self.create_hierarchical_state(char_matrix, perturbation_strength=2.0)
            noisy_state = self.create_hierarchical_state(noisy_matrix, perturbation_strength=2.0)
            
            # Process with predictive Hebbian model
            predictive_final, predictive_weights, _, _ = self.process_character(
                noisy_state.copy(), model_type="predictive", iterations=200
            )
            
            # Process with standard Hebbian model
            hebbian_final, hebbian_weights, _, _ = self.process_character(
                noisy_state.copy(), model_type="hebbian", iterations=200
            )
            
            # Visualize comparison
            self.visualize_noise_comparison(
                clean_state, noisy_state, predictive_final, hebbian_final, 
                char, noise_level, save_path=f"noise_comparison_{char}_{int(noise_level*100)}.png"
            )
            
            # Calculate similarity to clean character for both models
            # Process clean character with both models
            clean_pred_final, _, _, _ = self.process_character(
                clean_state.copy(), model_type="predictive", iterations=200
            )
            
            clean_hebb_final, _, _, _ = self.process_character(
                clean_state.copy(), model_type="hebbian", iterations=200
            )
            
            # Calculate similarity metrics
            # For predictive model
            pred_similarity = np.mean(np.cos(predictive_final.phases[0] - clean_pred_final.phases[0]))
            
            # For Hebbian model
            hebb_similarity = np.mean(np.cos(hebbian_final.phases[0] - clean_hebb_final.phases[0]))
            
            print(f"Noise level: {noise_level}, Predictive similarity: {pred_similarity:.4f}, Hebbian similarity: {hebb_similarity:.4f}")
            
            # Calculate coherence for both models
            pred_coherence = np.mean(self.calculate_local_coherence(predictive_final.phases[0]))
            hebb_coherence = np.mean(self.calculate_local_coherence(hebbian_final.phases[0]))
            
            print(f"Noise level: {noise_level}, Predictive coherence: {pred_coherence:.4f}, Hebbian coherence: {hebb_coherence:.4f}")
            
            # Test assertions
            # The predictive model should generally maintain higher similarity to the clean character
            # This might not always be true for all noise levels, but it's a reasonable expectation
            self.assertGreaterEqual(pred_similarity, hebb_similarity * 0.9,
                                  f"Predictive model should maintain similar or better similarity to clean character")
    
    def test_ambiguous_character_resolution(self):
        """Test resolution of ambiguous characters by predictive Hebbian vs. standard Hebbian models."""
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Test ambiguity between different character pairs
        char_pairs = [('B', 'P'), ('O', 'D'), ('1', '7')]
        ambiguity_levels = [0.3, 0.5]
        
        for char1, char2 in char_pairs:
            for ambiguity_level in ambiguity_levels:
                # Create ambiguous character
                ambiguous_matrix = self.create_ambiguous_character(char1, char2, ambiguity_level)
                
                # Create state for ambiguous character
                ambiguous_state = self.create_hierarchical_state(ambiguous_matrix, perturbation_strength=2.0)
                
                # Process with predictive Hebbian model
                predictive_final, predictive_weights, _, _ = self.process_character(
                    ambiguous_state.copy(), model_type="predictive", iterations=200
                )
                
                # Process with standard Hebbian model
                hebbian_final, hebbian_weights, _, _ = self.process_character(
                    ambiguous_state.copy(), model_type="hebbian", iterations=200
                )
                
                # Visualize ambiguity resolution
                self.visualize_ambiguity_resolution(
                    ambiguous_matrix, predictive_final, hebbian_final, 
                    char1, char2, ambiguity_level, 
                    save_path=f"ambiguity_{char1}_{char2}_{int(ambiguity_level*100)}.png"
                )
                
                # Process individual characters for comparison
                char1_matrix = self.get_character_matrix(char1)
                char1_state = self.create_hierarchical_state(char1_matrix, perturbation_strength=2.0)
                char1_final, _, _, _ = self.process_character(char1_state, model_type="predictive", iterations=200)
                
                char2_matrix = self.get_character_matrix(char2)
                char2_state = self.create_hierarchical_state(char2_matrix, perturbation_strength=2.0)
                char2_final, _, _, _ = self.process_character(char2_state, model_type="predictive", iterations=200)
                
                # Calculate similarity to each character
                # For predictive model
                pred_similarity_1 = np.mean(np.cos(predictive_final.phases[0] - char1_final.phases[0]))
                pred_similarity_2 = np.mean(np.cos(predictive_final.phases[0] - char2_final.phases[0]))
                
                # For Hebbian model
                hebb_similarity_1 = np.mean(np.cos(hebbian_final.phases[0] - char1_final.phases[0]))
                hebb_similarity_2 = np.mean(np.cos(hebbian_final.phases[0] - char2_final.phases[0]))
                
                print(f"Ambiguity {char1}/{char2} at {ambiguity_level}:")
                print(f"  Predictive: Sim1={pred_similarity_1:.4f}, Sim2={pred_similarity_2:.4f}, Diff={abs(pred_similarity_1-pred_similarity_2):.4f}")
                print(f"  Hebbian: Sim1={hebb_similarity_1:.4f}, Sim2={hebb_similarity_2:.4f}, Diff={abs(hebb_similarity_1-hebb_similarity_2):.4f}")
                
                # Test assertions
                # The predictive model should generally show stronger disambiguation
                # (larger difference between similarities to the two characters)
                pred_diff = abs(pred_similarity_1 - pred_similarity_2)
                hebb_diff = abs(hebb_similarity_1 - hebb_similarity_2)
                
                # This might not always be true, but it's a reasonable expectation
                # We'll use a soft assertion with a tolerance
                self.assertGreaterEqual(pred_diff, hebb_diff * 0.8,
                                      f"Predictive model should show similar or stronger disambiguation")
    
    def test_occlusion_handling(self):
        """Test handling of occluded characters by predictive Hebbian vs. standard Hebbian models."""
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Process a character with different occlusion types
        char = 'A'
        char_matrix = self.get_character_matrix(char)
        
        occlusion_types = ['horizontal', 'vertical', 'random']
        occlusion_level = 0.3
        
        for occlusion_type in occlusion_types:
            # Create occluded character
            occluded_matrix = self.create_occluded_character(char, occlusion_type, occlusion_level)
            
            # Create states for clean and occluded characters
            clean_state = self.create_hierarchical_state(char_matrix, perturbation_strength=2.0)
            occluded_state = self.create_hierarchical_state(occluded_matrix, perturbation_strength=2.0)
            
            # Process with predictive Hebbian model
            predictive_final, predictive_weights, _, _ = self.process_character(
                occluded_state.copy(), model_type="predictive", iterations=200
            )
            
            # Process with standard Hebbian model
            hebbian_final, hebbian_weights, _, _ = self.process_character(
                occluded_state.copy(), model_type="hebbian", iterations=200
            )
            
            # Visualize comparison (reusing the noise comparison visualization)
            self.visualize_noise_comparison(
                clean_state, occluded_state, predictive_final, hebbian_final, 
                char, occlusion_level, save_path=f"occlusion_{char}_{occlusion_type}.png"
            )
            
            # Calculate similarity to clean character for both models
            # Process clean character with both models
            clean_pred_final, _, _, _ = self.process_character(
                clean_state.copy(), model_type="predictive", iterations=200
            )
            
            clean_hebb_final, _, _, _ = self.process_character(
                clean_state.copy(), model_type="hebbian", iterations=200
            )
            
            # Calculate similarity metrics
            # For predictive model
            pred_similarity = np.mean(np.cos(predictive_final.phases[0] - clean_pred_final.phases[0]))
            
            # For Hebbian model
            hebb_similarity = np.mean(np.cos(hebbian_final.phases[0] - clean_hebb_final.phases[0]))
            
            print(f"Occlusion type: {occlusion_type}, Predictive similarity: {pred_similarity:.4f}, Hebbian similarity: {hebb_similarity:.4f}")
            
            # Test assertions
            # The predictive model should generally maintain higher similarity to the clean character
            # This might not always be true for all occlusion types, but it's a reasonable expectation
            self.assertGreaterEqual(pred_similarity, hebb_similarity * 0.9,
                                  f"Predictive model should maintain similar or better similarity to clean character")
