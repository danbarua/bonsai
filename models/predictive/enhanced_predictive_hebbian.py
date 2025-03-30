from dataclasses import dataclass, field
from typing import Any
import numpy as np
from numpy.typing import NDArray

from dynamics import LayeredOscillatorState, StateMutation

@dataclass
class EnhancedPredictiveHebbianOperator(StateMutation[LayeredOscillatorState]):
    """
    Enhanced implementation combining predictive coding between layers with 
    Hebbian-Kuramoto dynamics within layers, with additional features for 
    stability and analysis.
    """
    # Predictive coding parameters
    pc_learning_rate: float = 0.01
    pc_error_scaling: float = 0.5
    pc_precision: float = 1.0  # Inverse variance, controls confidence in predictions
    
    # Hebbian parameters
    hebb_learning_rate: float = 0.01
    hebb_decay_rate: float = 0.1
    
    # Common parameters
    dt: float = 0.1
    weight_normalization: bool = True  # Whether to normalize weights
    
    # State variables
    between_layer_weights: list[NDArray[np.float64]] = field(default_factory=list)
    within_layer_weights: list[NDArray[np.float64]] = field(default_factory=list)
    prediction_history: list[list[NDArray[np.float64]]] = field(default_factory=list)
    error_history: list[list[NDArray[np.float64]]] = field(default_factory=list)
    last_delta: dict[str, Any] = field(default_factory=dict)
    
    def apply(self, state: LayeredOscillatorState) -> LayeredOscillatorState:
        new_state = state.copy()
        layer_count = state.num_layers
        
        # Initialize weights and history if needed
        if not self.between_layer_weights:
            self._initialize_weights(state)
            self.prediction_history = [[] for _ in range(layer_count-1)]
            self.error_history = [[] for _ in range(layer_count-1)]
        
        # 1. Compute hierarchical predictions and errors (predictive coding)
        predictions, errors = self._compute_hierarchical_predictions(state)
        
        # Store for analysis
        for i in range(layer_count-1):
            self.prediction_history[i].append(predictions[i])
            self.error_history[i].append(errors[i])
            # Keep history bounded
            if len(self.prediction_history[i]) > 100:
                self.prediction_history[i].pop(0)
                self.error_history[i].pop(0)
        
        # 2. Compute phase updates from both mechanisms
        phase_updates = self._compute_combined_updates(state, errors)
        
        # 3. Apply updates to phases
        for i in range(layer_count):
            new_state._phases[i] = (state.phases[i] + self.dt * phase_updates[i]) % (2 * np.pi)
        
        # 4. Update weights according to their respective rules
        self._update_hebbian_weights(state)
        self._update_pc_weights(state, errors)
        
        # 5. Apply weight normalization if enabled
        if self.weight_normalization:
            self._normalize_weights()
        
        # 6. Collect metrics and analyze dynamics
        self._collect_metrics(state, new_state, predictions, errors)
        
        return new_state
    
    def _initialize_weights(self, state: LayeredOscillatorState) -> None:
        """Initialize weights with spectral initialization for better stability"""
        layer_count = state.num_layers
        
        # Between-layer weights (predictive coding)
        self.between_layer_weights = []
        for i in range(layer_count - 1):
            input_size = np.prod(state.layer_shapes[i])
            output_size = np.prod(state.layer_shapes[i+1])
            
            # Use spectral initialization (Xavier/Glorot-like)
            scale = np.sqrt(6.0 / (input_size + output_size))
            w = np.random.uniform(-scale, scale, (output_size, input_size))
            self.between_layer_weights.append(w)
        
        # Within-layer weights (Hebbian) - initialize near fixed point
        self.within_layer_weights = []
        for i in range(layer_count):
            n_oscillators = np.prod(state.layer_shapes[i])
            
            # Start near the theoretical fixed point with small noise
            phases_flat = state.phases[i].flatten()
            phase_diffs = phases_flat[:, np.newaxis] - phases_flat[np.newaxis, :]
            w = np.cos(phase_diffs) / self.hebb_decay_rate
            w += np.random.normal(0, 0.01, (n_oscillators, n_oscillators))
            
            self.within_layer_weights.append(w)
    
    def _compute_hierarchical_predictions(self, state: LayeredOscillatorState) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
        """Compute predictions and errors through the hierarchy"""
        layer_count = state.num_layers
        predictions = []
        errors = []
        
        # Process each layer pair
        for i in range(layer_count - 1):
            # Get activities (using complex representation for better gradient properties)
            lower_activity = np.exp(1j * state.phases[i]).flatten()
            higher_activity = np.exp(1j * state.phases[i+1]).flatten()
            
            # Project from lower to higher (prediction)
            prediction_complex = self.between_layer_weights[i] @ lower_activity
            
            # Normalize to unit circle
            prediction_norm = np.abs(prediction_complex)
            prediction_complex = prediction_complex / (prediction_norm + 1e-10)
            
            # Compute phase prediction and actual target
            prediction = np.angle(prediction_complex)
            target = np.angle(higher_activity)
            
            # Compute circular error (respecting phase periodicity)
            error = np.angle(np.exp(1j * (target - prediction)))
            
            # Scale by precision
            weighted_error = error * self.pc_precision
            
            predictions.append(prediction)
            errors.append(weighted_error)
        
        return predictions, errors
    
    def _compute_combined_updates(self, state: LayeredOscillatorState, 
                                 errors: list[NDArray[np.float64]]) -> list[NDArray[np.float64]]:
        """Compute combined phase updates from both mechanisms"""
        layer_count = state.num_layers
        updates = []
        
        for i in range(layer_count):
            shape = state.layer_shapes[i]
            phases_flat = state.phases[i].flatten()
            
            # 1. Hebbian-Kuramoto update (within-layer)
            phase_diffs = phases_flat[:, np.newaxis] - phases_flat[np.newaxis, :]
            sin_diffs = np.sin(phase_diffs)
            
            # Compute Kuramoto coupling
            hebbian_update = np.sum(self.within_layer_weights[i] * sin_diffs, axis=1)
            
            # Add natural frequencies
            hebbian_update += state.frequencies[i].flatten() * 2 * np.pi
            
            # 2. Predictive coding update (between-layer)
            pc_update = np.zeros_like(hebbian_update)
            
            # Bottom-up error from layer below
            if i > 0:
                # Project error from layer below
                bottom_up_error = self.between_layer_weights[i-1].T @ errors[i-1]
                pc_update += self.pc_error_scaling * bottom_up_error
            
            # Top-down error to layer above
            if i < layer_count - 1:
                # Direct influence of current error
                pc_update -= self.pc_error_scaling * errors[i].reshape(-1)
            
            # Combine updates and reshape
            combined_update = hebbian_update + pc_update
            updates.append(combined_update.reshape(shape))
        
        return updates
    
    def _update_hebbian_weights(self, state: LayeredOscillatorState) -> None:
        """Update within-layer weights according to Hebbian rule"""
        for i in range(state.num_layers):
            phases_flat = state.phases[i].flatten()
            
            # Compute phase differences and cosine (Hebbian term)
            phase_diffs = phases_flat[:, np.newaxis] - phases_flat[np.newaxis, :]
            cos_diffs = np.cos(phase_diffs)
            
            # Compute weight update (approach the fixed point solution)
            target_weights = cos_diffs / self.hebb_decay_rate
            weight_error = target_weights - self.within_layer_weights[i]
            
            # Apply updates with learning rate
            self.within_layer_weights[i] += self.dt * self.hebb_learning_rate * weight_error
    
    def _update_pc_weights(self, state: LayeredOscillatorState, errors: list[NDArray[np.float64]]) -> None:
        """Update between-layer weights according to predictive coding rule"""
        for i in range(state.num_layers - 1):
            # Use complex representation for better gradient properties
            lower_activity = np.exp(1j * state.phases[i]).flatten()
            
            # Compute weight update using error and activity
            # Using outer product for associative learning
            weight_update = self.pc_learning_rate * np.outer(
                errors[i] * 1j * np.exp(1j * errors[i]),  # Error gradient
                np.conj(lower_activity)                   # Input activity
            )
            
            # Extract real component and apply update
            self.between_layer_weights[i] += weight_update.real
    
    def _normalize_weights(self) -> None:
        """Apply normalization to prevent weight explosion"""
        # Normalize within-layer (Hebbian) weights
        for i in range(len(self.within_layer_weights)):
            # Apply row-wise normalization (preserves relative strengths)
            norms = np.sqrt(np.sum(self.within_layer_weights[i]**2, axis=1, keepdims=True))
            norms = np.maximum(norms, 1e-10)  # Avoid division by zero
            self.within_layer_weights[i] = self.within_layer_weights[i] / norms
        
        # Normalize between-layer (PC) weights
        for i in range(len(self.between_layer_weights)):
            # Use spectral normalization (helps with learning stability)
            u, s, vh = np.linalg.svd(self.between_layer_weights[i], full_matrices=False)
            max_sv = s[0]
            if max_sv > 1.0:
                self.between_layer_weights[i] = self.between_layer_weights[i] / max_sv
    
    def _collect_metrics(self, state: LayeredOscillatorState, new_state: LayeredOscillatorState,
                         predictions: list[NDArray[np.float64]], errors: list[NDArray[np.float64]]) -> None:
        """Collect comprehensive metrics for monitoring and analysis"""
        layer_count = state.num_layers
        
        # Phase coherence metrics
        coherence_values = []
        for i in range(layer_count):
            z = np.exp(1j * new_state.phases[i].flatten())
            coherence = float(np.abs(np.mean(z)))
            coherence_values.append(coherence)
        
        # Error metrics
        error_norms = [float(np.linalg.norm(err)) for err in errors]
        
        # Spectral analysis of weight matrices
        spectral_stats = self._analyze_weight_spectrum()
        
        # Fixed point analysis for Hebbian weights
        fixed_point_analysis = self._analyze_fixed_points(state)
        
        # Energy metrics
        energy = self._compute_system_energy(state)
        
        self.last_delta = {
            "type": "enhanced_predictive_hebbian",
            "coherence": coherence_values,
            "mean_coherence": float(np.mean(coherence_values)),
            "prediction_errors": error_norms,
            "total_error": float(np.sum(error_norms)),
            "weight_spectrum": spectral_stats,
            "fixed_point_analysis": fixed_point_analysis,
            "system_energy": energy
        }
    
    def _analyze_weight_spectrum(self) -> dict[str, Any]:
        """Analyze spectral properties of weight matrices"""
        spectrum_data = {
            "hebbian": [],
            "predictive": []
        }
        
        # Analyze Hebbian weights
        for i, w in enumerate(self.within_layer_weights):
            try:
                # Get top eigenvalues
                eigvals = np.linalg.eigvals(w)
                max_eigval = float(np.max(np.abs(eigvals)))
                min_eigval = float(np.min(np.abs(eigvals)))
                
                spectrum_data["hebbian"].append({
                    "layer": i,
                    "max_eigval": max_eigval,
                    "min_eigval": min_eigval,
                    "condition_number": max_eigval / (min_eigval + 1e-10)
                })
            except:
                # Handle numerical issues
                spectrum_data["hebbian"].append({
                    "layer": i,
                    "error": "SVD computation failed"
                })
        
        # Analyze predictive coding weights
        for i, w in enumerate(self.between_layer_weights):
            try:
                # Get singular values
                u, s, vh = np.linalg.svd(w, full_matrices=False)
                
                spectrum_data["predictive"].append({
                    "layers": f"{i}->{i+1}",
                    "max_sv": float(s[0]),
                    "min_sv": float(s[-1]),
                    "condition_number": float(s[0] / (s[-1] + 1e-10))
                })
            except:
                spectrum_data["predictive"].append({
                    "layers": f"{i}->{i+1}",
                    "error": "SVD computation failed"
                })
        
        return spectrum_data
    
    def _analyze_fixed_points(self, state: LayeredOscillatorState) -> dict[str, Any]:
        """Analyze how close the system is to theoretical fixed points"""
        fixed_point_data = []
        
        for i in range(state.num_layers):
            phases_flat = state.phases[i].flatten()
            
            # Compute current phase differences
            phase_diffs = phases_flat[:, np.newaxis] - phases_flat[np.newaxis, :]
            
            # Theoretical fixed point weights
            theoretical_weights = np.cos(phase_diffs) / self.hebb_decay_rate
            
            # Compute distance to fixed point
            weight_diff = theoretical_weights - self.within_layer_weights[i]
            distance = float(np.linalg.norm(weight_diff) / np.linalg.norm(theoretical_weights))
            
            fixed_point_data.append({
                "layer": i,
                "distance_to_fixed_point": distance,
                "max_deviation": float(np.max(np.abs(weight_diff))),
                "mean_deviation": float(np.mean(np.abs(weight_diff)))
            })
        
        return fixed_point_data
    
    def _compute_system_energy(self, state: LayeredOscillatorState) -> dict[str, float]:
        """Compute energy-based metrics for the system"""
        # Define system energy components
        hebbian_energy = 0.0
        pc_energy = 0.0
        
        # Compute Hebbian energy (negative of coupling satisfaction)
        for i in range(state.num_layers):
            phases_flat = state.phases[i].flatten()
            phase_diffs = phases_flat[:, np.newaxis] - phases_flat[np.newaxis, :]
            
            # Negative coupling energy (higher means more satisfied couplings)
            layer_energy = -np.sum(self.within_layer_weights[i] * np.cos(phase_diffs))
            hebbian_energy += float(layer_energy)
        
        # Compute predictive coding energy (prediction error)
        for i in range(state.num_layers - 1):
            lower_activity = np.exp(1j * state.phases[i]).flatten()
            higher_activity = np.exp(1j * state.phases[i+1]).flatten()
            
            prediction = self.between_layer_weights[i] @ lower_activity
            prediction_norm = np.abs(prediction)
            normalized_prediction = prediction / (prediction_norm + 1e-10)
            
            # Prediction error energy
            error = np.abs(higher_activity - normalized_prediction)
            layer_energy = float(np.sum(error**2))
            pc_energy += layer_energy
        
        return {
            "hebbian_energy": hebbian_energy,
            "pc_energy": pc_energy,
            "total_energy": hebbian_energy + pc_energy
        }
    
    def get_delta(self) -> dict[str, Any]:
        return self.last_delta
    
    def get_trajectory_analysis(self) -> dict[str, Any]:
        """Analyze learning trajectories over time"""
        layer_count = len(self.within_layer_weights)
        
        analysis = {
            "error_convergence": [],
            "weight_stability": [],
            "prediction_improvement": []
        }
        
        # Analyze error convergence
        for i in range(layer_count - 1):
            if len(self.error_history[i]) < 2:
                continue
                
            # Compute error reduction over time
            error_norms = [np.linalg.norm(err) for err in self.error_history[i]]
            error_reduction = (error_norms[0] - error_norms[-1]) / (error_norms[0] + 1e-10)
            
            analysis["error_convergence"].append({
                "layers": f"{i}->{i+1}",
                "initial_error": float(error_norms[0]),
                "final_error": float(error_norms[-1]),
                "reduction": float(error_reduction),
                "converged": float(error_norms[-1]) < 0.1
            })
        
        return analysis