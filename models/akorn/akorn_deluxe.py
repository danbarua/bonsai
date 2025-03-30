import numpy as np
from collections import deque
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
from numpy.typing import NDArray
from dynamics import StateMutation, LayeredOscillatorState

@dataclass
class DeluxeHebbianKuramotoOperator(StateMutation[LayeredOscillatorState]):
    """
    Implements Hebbian-Kuramoto dynamics with self-organizing intelligence.
    """
    dt: float = 0.1
    alpha: float = 0.1  # Weight decay parameter
    mu: float = 0.01    # Hebbian learning rate
    oscillator_dim: int = 4 # Dimensionality of oscillators
    grid_size: Tuple[int, int] = (16, 16) # Size of the oscillator grid
    weight_symmetry: bool = False # ADDED: Control weight symmetry
    
    # Configuration parameters (can be adjusted)
    config: Dict[str, Any] = field(default_factory=lambda: {
        "LOCAL_RANGE": 2,
        "LONG_RANGE_PROB": 0.05,
        "LONG_RANGE_SCALE": 10,
        "LONG_RANGE_STRENGTH": 0.5,
        "NUM_HUBS": 4,
        "HUB_FACTOR": 2.0,
        "RESONANCE_SENSITIVITY": 0.1,
        "RESONANCE_LEARNING_RATE": 0.01,
        "RESONANCE_DECAY": 0.001,
        "STABILITY_WINDOW": 10,
        "STABILITY_THRESHOLD": 0.01,
        "NOVELTY_THRESHOLD": 0.2,
        "RELATIONSHIP_LEARNING_RATE": 0.1,
        "RELATIONSHIP_DECAY": 0.01,
        "HISTORY_LENGTH": 50
    })
    
    # Internal state variables
    coupling_matrix: NDArray[np.float64] = field(init=False)
    freq: NDArray[np.float64] = field(init=False)
    W: NDArray[np.float64] = field(init=False)
    pattern_memory: List[Dict[str, Any]] = field(default_factory=list)
    pattern_relationships: NDArray[np.float64] = field(init=False)
    phase_history: deque = field(default_factory=lambda: deque(maxlen=50))
    
    weights: List[NDArray[np.float64]] = field(default_factory=list)
    equivalent_kuramoto_states: List[NDArray[np.float64]] = field(default_factory=list)
    last_delta: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Initialize coupling matrix, frequencies, and other internal state
        self.coupling_matrix = self.initialize_connectome_inspired_coupling()
        self.freq = np.random.uniform(0.1, 1.0, size=self.grid_size)  # Example frequencies
        self.W = np.zeros((self.oscillator_dim, self.oscillator_dim)) # Cross-dimensional coupling
        self.pattern_relationships = np.zeros((0, 0))
        self.phase_history = deque(maxlen=self.config["HISTORY_LENGTH"])
    
    def initialize_connectome_inspired_coupling(self) -> NDArray[np.float64]:
        """
        Initializes the coupling matrix with connectome-inspired properties.
        """
        # Create distance matrix based on grid positions
        y_coords, x_coords = np.meshgrid(range(self.grid_size[0]), range(self.grid_size[1]), indexing='ij')
        positions = np.stack([y_coords, x_coords], axis=-1).reshape(-1, 2)
        
        distances = np.sqrt(((positions[:, None, :] - positions[None, :, :]) ** 2).sum(axis=2))
        
        # Short-range dense connections, long-range sparse connections
        short_range = np.exp(-distances**2 / (2 * self.config["LOCAL_RANGE"]**2))
        long_range = (np.random.rand(*distances.shape) < (self.config["LONG_RANGE_PROB"] * np.exp(-distances / self.config["LONG_RANGE_SCALE"])))
        
        # Combine into coupling matrix
        coupling = short_range + self.config["LONG_RANGE_STRENGTH"] * long_range
        
        # Add hub structure
        coupling = self.add_hub_structure(coupling)
        
        return coupling.reshape(self.grid_size + self.grid_size)
    
    def add_hub_structure(self, coupling: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Adds hub-based connectivity to the coupling matrix.
        """
        # Identify hub locations (could be fixed or emergent)
        hub_indices = np.random.choice(
            np.prod(self.grid_size), 
            size=self.config["NUM_HUBS"], 
            replace=False
        )
        
        hub_mask = np.zeros(np.prod(self.grid_size), dtype=bool)
        hub_mask[hub_indices] = True
        hub_mask = hub_mask.reshape(self.grid_size)
        
        # Enhance connectivity to/from hubs
        for h_idx in hub_indices:
            h_y, h_x = np.unravel_index(h_idx, self.grid_size)
            # Increase coupling to and from this hub
            #coupling[h_y, h_x, :, :] *= self.config["HUB_FACTOR"]
            #coupling[:, :, h_y, h_x] *= self.config["HUB_FACTOR"]

            coupling[h_y, h_x] *= self.config["HUB_FACTOR"]
            coupling[h_y, h_x] *= self.config["HUB_FACTOR"]
        
        return coupling
    
    def detect_harmonic_relationships(self) -> NDArray[np.float64]:
        """
        Detects harmonic relationships between oscillator frequencies.
        """
        # Reshape frequencies for easier calculation
        freqs_flat = self.freq.reshape(-1)
        
        # Compute all pairwise frequency ratios
        ratios = freqs_flat[:, None] / (freqs_flat[None, :] + 1e-10)
        
        # Find near-integer and simple fraction ratios (like 2:1, 3:2, etc.)
        target_ratios = np.array([1.0, 2.0, 3.0, 1.5, 4.0, 1.25, 1.33, 1.67])
        
        resonance = np.zeros_like(ratios)
        for target in target_ratios:
            resonance += np.exp(-(ratios - target)**2 / (2 * self.config["RESONANCE_SENSITIVITY"]**2))
        
        return resonance
    
    def update_coupling_from_resonance(self):
        """
        Updates the cross-dimensional coupling matrix based on harmonic resonance.
        """
        resonance = self.detect_harmonic_relationships()
        
        # Use resonance to modulate cross-dimensional coupling
        # Average resonance across the grid
        mean_resonance = np.mean(resonance)
        # Update W matrix based on resonance
        self.W += self.config["RESONANCE_LEARNING_RATE"] * (
            mean_resonance - self.W
        )
        # Apply decay to keep matrix sparse
        self.W *= (1 - self.config["RESONANCE_DECAY"])
    
    def phase_distance(self, phase1: NDArray[np.float64], phase2: NDArray[np.float64]) -> float:
        """
        Calculates the distance between two phase patterns.
        """
        diff = np.abs(phase1 - phase2)
        return float(np.mean(diff))
    
    def detect_stable_representations(self) -> bool:
        """
        Detects when the system forms stable phase representations.
        """
        # Check if phase patterns have been stable for several steps
        if len(self.phase_history) < self.config["STABILITY_WINDOW"]:
            return False
        
        recent_phases = list(self.phase_history)[-self.config["STABILITY_WINDOW"]:]
        differences = []
        
        for i in range(1, len(recent_phases)):
            diff = self.phase_distance(recent_phases[i], recent_phases[i-1])
            differences.append(diff)
        
        avg_difference = np.mean(differences)
        return avg_difference < self.config["STABILITY_THRESHOLD"]
    
    def extract_pattern_representation(self) -> NDArray[np.float64]:
        """
        Extracts a representation of the current phase pattern.
        """
        # Flatten the phase pattern for simplicity
        return self.phases.flatten()
    
    def pattern_distance(self, pattern1: NDArray[np.float64], pattern2: NDArray[np.float64]) -> float:
        """
        Calculates the distance between two pattern representations.
        """
        return float(np.linalg.norm(pattern1 - pattern2))
    
    def store_new_pattern(self, pattern_representation: NDArray[np.float64]):
        """
        Stores a new pattern in memory.
        """
        # Create new pattern entry
        pattern_idx = len(self.pattern_memory)
        self.pattern_memory.append({
            'representation': pattern_representation,
            'activation_count': 1,
            'last_activation': time.time(),
            # 'associated_context': self.extract_context() # Implement context extraction
        })
        
        # Expand relationship matrix
        old_size = self.pattern_relationships.shape[0]
        new_relationships = np.zeros((old_size+1, old_size+1))
        if old_size > 0:
            new_relationships[:old_size, :old_size] = self.pattern_relationships
        self.pattern_relationships = new_relationships
    
    def update_pattern_relationships(self, pattern_idx1: int, pattern_idx2: int):
        """
        Strengthens the connection between patterns that activate in sequence.
        """
        # Strengthen connection between patterns that activate in sequence
        self.pattern_relationships[pattern_idx1, pattern_idx2] += self.config["RELATIONSHIP_LEARNING_RATE"]
        # Apply decay to keep matrix sparse
        self.pattern_relationships *= (1 - self.config["RELATIONSHIP_DECAY"])
    
    def apply(self, state: LayeredOscillatorState) -> LayeredOscillatorState:
        new_state = state.copy()
        layer_count = state.num_layers
        
        # Initialize if needed
        if not self.weights:
            self.weights = []
            self.equivalent_kuramoto_states = []
            for i in range(layer_count):
                shape = state.phases[i].shape
                n_oscillators = np.prod(shape)
                self.weights.append(self.coupling_matrix) # Use connectome-inspired coupling
                self.equivalent_kuramoto_states.append(np.random.rand(n_oscillators, self.oscillator_dim))
                self.equivalent_kuramoto_states[i] /= np.linalg.norm(self.equivalent_kuramoto_states[i], axis=1, keepdims=True) # Normalize
                new_state.phases[i] = np.random.rand(n_oscillators, self.oscillator_dim)
                new_state.phases[i] /= np.linalg.norm(new_state.phases[i], axis=1, keepdims=True) # Normalize
                new_state.phases[i] = new_state.phases[i].reshape(shape + (self.oscillator_dim,))
                self.equivalent_kuramoto_states[i] = self.equivalent_kuramoto_states[i].reshape(shape + (self.oscillator_dim,))
        
        # For each layer, update both actual state and equivalent Kuramoto state
        for i in range(layer_count):
            # Get flattened phases
            phases_flat = state.phases[i].reshape(-1, self.oscillator_dim)
            equiv_phases_flat = self.equivalent_kuramoto_states[i].reshape(-1, self.oscillator_dim)
            shape = state.phases[i].shape[:-1] # Shape without oscillator_dim
            n_oscillators = phases_flat.shape[0]
            
            # Update equivalent Kuramoto state (with fixed weights = 1/(2α))
            # This gives us stability information
            equiv_diffs = equiv_phases_flat[:, np.newaxis, :] - equiv_phases_flat[np.newaxis, :, :] # (N, N, dim)
            equiv_update = state.frequencies[i].flatten() * 2 * np.pi + self.freq[i] @ equiv_phases_flat # (N, dim)
            equiv_update += np.sum(np.sin(equiv_diffs) / (2 * self.alpha), axis=1) # (N, dim)
            equiv_new = (equiv_phases_flat + self.dt * equiv_update) # (N, dim)
            equiv_new /= np.linalg.norm(equiv_new, axis=1, keepdims=True) # Normalize
            
            # Update actual Hebbian-Kuramoto state
            # The fixed point is at half the phase angle
            actual_update = state.frequencies[i].flatten() * 2 * np.pi #+ self.natural_frequencies[i] @ phases_flat # (N, dim)
            
            # Compute phase differences
            phase_diffs = phases_flat[:, np.newaxis, :] - phases_flat[np.newaxis, :, :] # (N, N, dim)
            
            # Get current weights from cosine of phase differences (fixed point relation)
            # Dot product is a more direct measure of vector similarity
            dot_products = np.sum(phases_flat[:, None, :] * phases_flat[None, :, :], axis=2)
            # Normalize to account for potentially non-unit vectors
            norms = np.linalg.norm(phases_flat, axis=1)
            normalized_dot = dot_products / (norms[:, None] * norms[None, :] + 1e-10)
            current_weights = normalized_dot / self.alpha
            
            # Enforce weight symmetry (or not)
            if not self.weight_symmetry:
                self.weights[i] = current_weights
            else:
                self.weights[i] = (current_weights + current_weights.T) / 2
            
            # Project to tangent space
            actual_update = self.project_to_tangent_space(phases_flat, actual_update)
            
            # Use these weights with sin term for phase update
            actual_update += np.sum(self.weights[i][:, :, np.newaxis] * np.sin(phase_diffs), axis=1) # (N, dim)
            actual_new = (phases_flat + self.dt * actual_update) # (N, dim)
            actual_new /= np.linalg.norm(actual_new, axis=1, keepdims=True) # Normalize
            
            # Store updated states
            self.equivalent_kuramoto_states[i] = equiv_new.reshape(shape + (self.oscillator_dim,))
            new_state._phases[i] = actual_new.reshape(shape + (self.oscillator_dim,))
            
        # Discover patterns
        self.discover_patterns()
        
        # Update coupling from resonance
        self.update_coupling_from_resonance()
        
        # Store phase history
        self.phase_history.append(new_state.phases[i])
        
        # Compute metrics
        coherence_values = []
        for i in range(layer_count):
            # MODIFIED: Calculate coherence using vector states
            z = new_state.phases[i].reshape(-1, self.oscillator_dim)
            coherence = self.calculate_vector_coherence(z)
            coherence_values.append(coherence)
        
        # Check stability using equivalent Kuramoto dynamics
        stability_metrics = self._analyze_stability(new_state.phases[i])
        
        self.last_delta = {
            "type": "enhanced_hebbian_kuramoto",
            "coherence": coherence_values,
            "mean_coherence": float(np.mean(coherence_values)),
            "stability": stability_metrics,
            "num_patterns": len(self.pattern_memory) # Track pattern count
        }
        
        return new_state
    
    def _analyze_stability(self, phases) -> Dict[str, Any]:
        """Analyze stability using the equivalent Kuramoto model"""
        # This would implement the stability analysis based on the paper's findings
        # For now just return placeholder metrics
        # MODIFIED: Calculate a more meaningful stability metric
        # Example: Variance of phase differences
        phase_diffs = phases[:, np.newaxis] - phases[np.newaxis, :]
        phase_variance = np.var(phase_diffs)
        return {
            "phase_variance": float(phase_variance)
        }
    
    def get_delta(self) -> Dict[str, Any]:
        return self.last_delta
    
    def calculate_vector_coherence(self, vectors: NDArray[np.float64]) -> float:
        """
        Calculates the coherence of a set of multi-dimensional vectors.
        """
        # For unit vectors, this measures alignment
        mean_vector = np.mean(vectors, axis=0)
        coherence = np.linalg.norm(mean_vector)
        return coherence
    
    def project_to_tangent_space(self, state_vectors: NDArray[np.float64], update_vectors: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Projects update vectors onto the tangent space of the state vectors.
        """
        # Remove component parallel to current state vector
        parallel_component = np.sum(
            state_vectors * update_vectors, 
            axis=1, 
            keepdims=True
        ) * state_vectors
        
        projection = update_vectors - parallel_component
        return projection
    
    def make_skew_symmetric(self, matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        """Convert a matrix to skew-symmetric form (A^T = -A)"""
        return 0.5 * (matrix - matrix.transpose(0, 2, 1))