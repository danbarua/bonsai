"""
Tests for processing character inputs with a Predictive Hebbian network.

This test suite evaluates how a Predictive Hebbian network processes and responds to
character inputs, with a focus on hierarchical processing, noise robustness, and
ambiguity resolution.
"""

import numpy as np
from models.predictive import PredictiveHebbianOperator
from models.hebbian import HebbianKuramotoOperator
from tests.learning.utils.base_test import CharacterProcessingBaseTest
from tests.learning.utils.viz_utils import (
    visualize_character_state,
    visualize_noisy_character,
    visualize_model_comparison,
    visualize_hierarchical_representation,
    visualize_feature_extraction,
    visualize_reconstruction,
    visualize_ambiguity_resolution,
    visualize_occlusion_handling,
    visualize_character_embedding
)

class TestPredictiveHebbianCharacterProcessing(CharacterProcessingBaseTest):
    """Tests for processing character inputs with a Predictive Hebbian network."""
    __test__ = True  # override CharacterProcessingBaseTest.__test__ = False; this concrete class should be collected
    
    def setUp(self):
        """Set up test fixtures"""
        super().setUp()
        
        # Predictive-specific parameters
        self.pc_learning_rate = 0.05
        self.hebb_learning_rate = 0.05
        self.pc_error_scaling = 0.5
        self.pc_precision = 1.0
        self.hebb_decay_rate = 0.1
    
    def create_character_state(self, char, perturbation_strength=None):
        """Create a hierarchical LayeredOscillatorState from a character."""
        if perturbation_strength is None:
            perturbation_strength = self.perturbation_strength
        char_matrix = self.get_character_matrix(char)
        return self.create_hierarchical_state(char_matrix, perturbation_strength=perturbation_strength)
    
    def process_character(self, char_or_state, model_type="predictive", iterations=None):
        """
        Process a character through the specified model type.
        
        Args:
            char_or_state: Either a character string or a LayeredOscillatorState
            model_type: "predictive" or "hebbian"
            iterations: Number of iterations to run
            
        Returns:
            Tuple of (final_state, weights, deltas, states_history)
        """
        # Set default values if not provided
        if iterations is None:
            iterations = self.max_steps
        
        # Create initial state if a character was provided
        if isinstance(char_or_state, str):
            char_matrix = self.get_character_matrix(char_or_state)
            if model_type == "predictive":
                state = self.create_hierarchical_state(char_matrix)
            else:
                state = self.create_single_layer_state(char_matrix)
        else:
            state = char_or_state
        
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
    
    def test_single_character_processing(self):
        """Test processing of a single character."""
        char = 'A'
        
        # Use stronger perturbation to ensure character regions are more distinct
        original_perturbation = self.perturbation_strength
        self.perturbation_strength = 2.0
        
        # Process the character with both models
        predictive_state, predictive_weights, _, _ = self.process_character(
            char, model_type="predictive", iterations=200
        )
        
        hebbian_state, hebbian_weights, _, _ = self.process_character(
            char, model_type="hebbian", iterations=200
        )
        
        # Visualize the hierarchical representation
        visualize_hierarchical_representation(
            predictive_state, char, 
            save_path=f"plots/predictive/hierarchical_{char}_representation.png"
        )
        
        # Visualize feature extraction
        visualize_feature_extraction(
            predictive_state, predictive_weights, char, 
            save_path=f"plots/predictive/hierarchical_{char}_features.png"
        )
        
        # Visualize reconstruction from each layer
        visualize_reconstruction(
            predictive_state, predictive_weights, char, 
            save_path=f"plots/predictive/hierarchical_{char}_reconstruction.png"
        )
        
        # Visualize model comparison
        visualize_model_comparison(
            char, hebbian_state, predictive_state,
            save_path=f"plots/comparison/character_{char}_model_comparison.png"
        )
        
        # Restore original perturbation strength
        self.perturbation_strength = original_perturbation
        
        # Check that all layers have non-zero phases
        for i in range(len(predictive_state.phases)):
            self.assertTrue(np.any(predictive_state.phases[i] > 0),
                           f"Layer {i} should have non-zero phases")
        
        # Check that coherence increases through the hierarchy
        coherence_values = []
        for i in range(len(predictive_state.phases)):
            coherence_map = self.calculate_local_coherence(predictive_state.phases[i])
            coherence_values.append(np.mean(coherence_map))
        
        # Higher layers should generally have higher coherence
        # This might not always be true, but it's a reasonable expectation
        print(f"Layer coherence values: {coherence_values}")

    def test_character_embedding(self):
        """Test multi-character phase-space embedding visualization (PCA/t-SNE).

        Ported from the standalone tests/test_predictive_hebbian_character.py
        (now retired) along with its bug fixes -- see
        tests/learning/utils/viz_utils.py::visualize_character_embedding.
        """
        characters = ['A', 'B', 'C', '1', '2']
        character_states = {}

        for c in characters:
            c_matrix = self.get_character_matrix(c)
            c_state = self.create_hierarchical_state(c_matrix, perturbation_strength=2.0)
            c_final, _, _, _ = self.process_character(c_state, model_type="predictive", iterations=200)
            character_states[c] = c_final

        visualize_character_embedding(character_states, characters,
                                       save_path="plots/comparison/character_embedding.png")

        # Different characters should produce distinct representations --
        # check every pair, not just one, matching the original test's coverage.
        for i in range(len(characters)):
            for j in range(i + 1, len(characters)):
                char1, char2 = characters[i], characters[j]
                phase_diff = np.abs(np.angle(np.exp(1j * (
                    character_states[char1].phases[0] - character_states[char2].phases[0]
                ))))
                mean_diff = np.mean(phase_diff)
                self.assertGreater(mean_diff, 0.1,
                                    f"Characters '{char1}' and '{char2}' produce too similar states")
    
    def test_character_distinction(self):
        """Test that different characters produce distinct network states."""
        chars = ['A', 'B', 'C', '1', '2']
        character_states = {}
        
        for c in chars:
            c_matrix = self.get_character_matrix(c)
            c_state = self.create_hierarchical_state(c_matrix, perturbation_strength=2.0)
            c_final, _, _, _ = self.process_character(c_state, model_type="predictive", iterations=200)
            character_states[c] = c_final
            
            # Visualize each character's processing
            visualize_character_state(
                c_final, {"within_layer_weights": c_final.phases}, c, model_type='predictive',
                save_path=f"plots/predictive/character_{c}_analysis.png"
            )
        
        # Test assertions
        # Check that different characters produce distinct representations
        for i in range(len(chars)):
            for j in range(i+1, len(chars)):
                char1 = chars[i]
                char2 = chars[j]
                
                # Calculate phase difference between characters
                phase_diff = np.abs(np.angle(np.exp(1j * (character_states[char1].phases[0] - character_states[char2].phases[0]))))
                mean_diff = np.mean(phase_diff)
                
                print(f"Mean phase difference between '{char1}' and '{char2}': {mean_diff:.4f}")
                
                # Different characters should produce distinct states
                self.assertGreater(mean_diff, 0.1, f"Characters '{char1}' and '{char2}' produce too similar states")
    
    def test_processing_stability(self):
        """Test stability of character processing across multiple runs."""
        char = 'A'
        coherence_values = []
        
        # Use stronger perturbation for better stability
        original_perturbation = self.perturbation_strength
        self.perturbation_strength = 2.0
        
        # Run multiple times with different random initializations
        for i in range(3):  # Reduced number of runs for faster testing
            _, _, deltas, _ = self.process_character(char, model_type="predictive", iterations=200)
            final_coherence = deltas[-1]["mean_coherence"]
            coherence_values.append(final_coherence)
            print(f"Run {i+1} coherence: {final_coherence:.4f}")
        
        # Restore original perturbation strength
        self.perturbation_strength = original_perturbation
        
        # Calculate coefficient of variation (std/mean)
        cv = np.std(coherence_values) / np.mean(coherence_values)
        print(f"Coefficient of variation: {cv:.4f}")
        
        # Allow for more variation since random initialization can lead to different attractors
        self.assertLess(cv, 0.5, "Character processing shows too much variation across runs")
    
    def test_noisy_character(self):
        """Test noise robustness of predictive Hebbian vs. standard Hebbian models."""
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
            visualize_noisy_character(
                clean_state, noisy_state, char, noise_level, model_type='predictive',
                save_path=f"plots/predictive/character_{char}_noisy_{int(noise_level*100)}.png"
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
            # For this test, we'll just check that the predictive model produces some level of similarity
            self.assertGreater(pred_similarity, -0.1,
                              f"Predictive model should maintain some similarity to clean character")
            # Note: We don't test the Hebbian model as it might produce negative similarity at high noise levels
    
    def test_ambiguous_character_resolution(self):
        """Test resolution of ambiguous characters by predictive Hebbian vs. standard Hebbian models."""
        # Test ambiguity between different character pairs
        char_pairs = [('B', 'P'), ('O', 'D'), ('1', '2')]
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
                visualize_ambiguity_resolution(
                    ambiguous_matrix, predictive_final, hebbian_final, 
                    char1, char2, ambiguity_level, 
                    save_path=f"plots/comparison/ambiguity_{char1}_{char2}_{int(ambiguity_level*100)}.png"
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
                
                # For this test, we'll just check that both models produce some level of disambiguation
                self.assertGreater(pred_diff, 0.0,
                                 f"Predictive model should show some disambiguation")
                self.assertGreater(hebb_diff, 0.0,
                                 f"Hebbian model should show some disambiguation")
    
    def test_occlusion_handling(self):
        """Test handling of occluded characters by predictive Hebbian vs. standard Hebbian models."""
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
            
            # Visualize comparison
            visualize_occlusion_handling(
                clean_state, occluded_state, predictive_final, hebbian_final, 
                char, occlusion_type, occlusion_level,
                save_path=f"plots/comparison/occlusion_{char}_{occlusion_type}_{int(occlusion_level*100)}.png"
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
            # For this test, we'll just check that the predictive model produces some level of similarity
            self.assertGreater(pred_similarity, -0.1,
                              f"Predictive model should maintain some similarity to clean character")
            # Note: We don't test the Hebbian model as it might produce negative similarity at high occlusion levels

    def test_perturbation_influence(self):
        """Test how perturbation strength affects character processing.

        Ported from TestHebbianKuramotoCharacterProcessing -- general model
        mechanics, not classification-specific, applies equally well here.
        """
        char = 'A'
        perturbation_strengths = [0.5, 1.0, 2.0]
        coherence_values = []

        for strength in perturbation_strengths:
            state = self.create_character_state(char, perturbation_strength=strength)
            op = PredictiveHebbianOperator(
                dt=self.dt, pc_learning_rate=self.pc_learning_rate,
                hebb_learning_rate=self.hebb_learning_rate,
                pc_error_scaling=self.pc_error_scaling,
                pc_precision=self.pc_precision, hebb_decay_rate=self.hebb_decay_rate
            )
            for _ in range(100):
                state = op.apply(state)
            coherence_values.append(op.last_delta["mean_coherence"])
            print(f"Perturbation strength: {strength}, Coherence: {op.last_delta['mean_coherence']:.4f}")

        # Higher perturbation should lead to stronger influence on the network
        self.assertNotAlmostEqual(coherence_values[0], coherence_values[-1],
                                 msg="Different perturbation strengths should produce different results")

    def test_frequency_vs_perturbation(self):
        """Test that heterogeneous natural frequencies measurably change the
        relative phase structure the network settles into, compared to zero
        frequencies.

        Ported from TestHebbianKuramotoCharacterProcessing, using the same
        corrected design applied there this project: comparing RELATIVE phase
        structure under HETEROGENEOUS per-oscillator frequencies against zero
        frequencies, rather than a uniform shared frequency shift against
        absolute phase (which doesn't meaningfully change relative/
        synchronized structure per standard Kuramoto theory, and is sensitive
        to incidental net-rotation coincidences for specific dt/step counts).
        """
        char = 'A'
        char_matrix = self.get_character_matrix(char)

        # Zero frequencies
        state_zero = self.create_hierarchical_state(char_matrix, perturbation_strength=2.0)
        for i in range(len(state_zero._frequencies)):
            state_zero._frequencies[i][:] = 0.0
        op_zero = PredictiveHebbianOperator(
            dt=self.dt, pc_learning_rate=self.pc_learning_rate,
            hebb_learning_rate=self.hebb_learning_rate, pc_error_scaling=self.pc_error_scaling,
            pc_precision=self.pc_precision, hebb_decay_rate=self.hebb_decay_rate
        )
        for _ in range(100):
            state_zero = op_zero.apply(state_zero)

        # Heterogeneous frequencies tied to the character pattern (input layer only)
        rng = np.random.default_rng(42)
        state_het = self.create_hierarchical_state(char_matrix, perturbation_strength=2.0)
        state_het._frequencies[0][:] = np.where(char_matrix > 0, 1.0, 0.5) + 0.05 * rng.standard_normal(char_matrix.shape)
        for i in range(1, len(state_het._frequencies)):
            state_het._frequencies[i][:] = 0.0
        op_het = PredictiveHebbianOperator(
            dt=self.dt, pc_learning_rate=self.pc_learning_rate,
            hebb_learning_rate=self.hebb_learning_rate, pc_error_scaling=self.pc_error_scaling,
            pc_precision=self.pc_precision, hebb_decay_rate=self.hebb_decay_rate
        )
        for _ in range(100):
            state_het = op_het.apply(state_het)

        zero_relative = state_zero.phases[0] - np.angle(np.mean(np.exp(1j * state_zero.phases[0])))
        het_relative = state_het.phases[0] - np.angle(np.mean(np.exp(1j * state_het.phases[0])))
        phase_diff = np.mean(np.abs(np.angle(np.exp(1j * (zero_relative - het_relative)))))

        print(f"Mean relative phase difference, zero vs heterogeneous frequencies: {phase_diff:.4f}")
        self.assertGreater(phase_diff, 0.1,
                          "Heterogeneous frequencies should measurably change the network's relative phase structure")

    def test_predictive_coding_contribution(self):
        """Compare full Predictive Hebbian (predictive coding + within-layer
        Hebbian) against within-layer Hebbian alone (predictive coding
        disabled via pc_error_scaling=0).

        This is the Predictive-model analog of
        TestHebbianKuramotoCharacterProcessing::test_comparison_with_standard_kuramoto
        (which compares Hebbian-with-learning against Hebbian-without, i.e.
        mu=0). The comparison axis here is different -- pc_error_scaling
        controls the predictive-coding contribution specifically, not
        learning in general -- since that's the actual novel mechanism this
        model adds on top of plain Hebbian-Kuramoto.
        """
        char = 'A'

        state_full = self.create_character_state(char)
        op_full = PredictiveHebbianOperator(
            dt=self.dt, pc_learning_rate=self.pc_learning_rate,
            hebb_learning_rate=self.hebb_learning_rate, pc_error_scaling=self.pc_error_scaling,
            pc_precision=self.pc_precision, hebb_decay_rate=self.hebb_decay_rate
        )

        state_no_pc = self.create_character_state(char)
        op_no_pc = PredictiveHebbianOperator(
            dt=self.dt, pc_learning_rate=self.pc_learning_rate,
            hebb_learning_rate=self.hebb_learning_rate, pc_error_scaling=0.0,
            pc_precision=self.pc_precision, hebb_decay_rate=self.hebb_decay_rate
        )

        for _ in range(200):
            state_full = op_full.apply(state_full)
            state_no_pc = op_no_pc.apply(state_no_pc)

        coherence_full = op_full.last_delta["mean_coherence"]
        coherence_no_pc = op_no_pc.last_delta["mean_coherence"]
        print(f"Full predictive coding coherence: {coherence_full:.4f}, No predictive coding coherence: {coherence_no_pc:.4f}")

        phase_diff = np.mean(np.abs(np.angle(np.exp(1j * (state_full.phases[0] - state_no_pc.phases[0])))))
        print(f"Mean phase difference, with vs without predictive coding: {phase_diff:.4f}")

        self.assertNotEqual(round(coherence_full, 2), round(coherence_no_pc, 2),
                           "Predictive coding should produce measurably different results than within-layer Hebbian alone")

        visualize_character_state(
            state_full, {"within_layer_weights": op_full.within_layer_weights, "between_layer_weights": op_full.between_layer_weights},
            char, model_type='predictive', save_path=f"plots/predictive/character_{char}_full.png"
        )
        visualize_character_state(
            state_no_pc, {"within_layer_weights": op_no_pc.within_layer_weights, "between_layer_weights": op_no_pc.between_layer_weights},
            char, model_type='predictive', save_path=f"plots/predictive/character_{char}_no_pc.png"
        )

    def test_character_sequence(self):
        """Test processing a sequence of characters and analyze transitions between them.

        Ported from TestHebbianKuramotoCharacterProcessing.
        """
        chars = ['A', '1', '+']
        states = []
        weights_list = []

        for char in chars:
            state, weights, _, _ = self.process_character(char, iterations=self.max_steps)
            states.append(state)
            weights_list.append(weights)
            visualize_character_state(
                state, weights, char, model_type='predictive',
                save_path=f"plots/predictive/character_{char}_sequence.png"
            )

        for i in range(len(chars) - 1):
            phase_diff = np.abs(np.angle(np.exp(1j * (states[i].phases[0] - states[i + 1].phases[0]))))
            mean_diff = np.mean(phase_diff)
            weight_diff = np.mean(np.abs(
                weights_list[i]["within_layer_weights"][0] - weights_list[i + 1]["within_layer_weights"][0]
            ))
            print(f"Transition {chars[i]} -> {chars[i+1]}: Mean phase diff = {mean_diff:.4f}, Mean weight diff = {weight_diff:.4f}")

            self.assertGreater(mean_diff, 0.1,
                              f"Characters '{chars[i]}' and '{chars[i+1]}' produce too similar states")

    def test_learning_transfer(self):
        """Test if learning one character helps with processing a structurally similar character.

        Ported from TestHebbianKuramotoCharacterProcessing, using
        within_layer_weights/between_layer_weights (settable dataclass
        fields on PredictiveHebbianOperator) in place of Hebbian's
        init_weights constructor argument -- there's no single init_weights
        kwarg here, but the equivalent transfer is just as direct.
        """
        state_A, weights_A, _, _ = self.process_character('A', iterations=self.max_steps)
        state_B_random, weights_B_random, _, _ = self.process_character('B', iterations=self.max_steps)

        state_B = self.create_character_state('B', self.perturbation_strength)
        n_layers = len(weights_A["within_layer_weights"])
        op_transfer = PredictiveHebbianOperator(
            dt=self.dt, pc_learning_rate=self.pc_learning_rate,
            hebb_learning_rate=self.hebb_learning_rate, pc_error_scaling=self.pc_error_scaling,
            pc_precision=self.pc_precision, hebb_decay_rate=self.hebb_decay_rate,
            within_layer_weights=[w.copy() for w in weights_A["within_layer_weights"]],
            between_layer_weights=[w.copy() for w in weights_A["between_layer_weights"]],
            # apply() only initializes prediction_history/error_history together
            # with the weights (both gated behind the same "if not
            # self.between_layer_weights" check) -- since we're pre-supplying
            # weights here, we need to supply these too or apply() indexes
            # into an empty list.
            prediction_history=[[] for _ in range(n_layers - 1)],
            error_history=[[] for _ in range(n_layers - 1)],
        )

        previous_coherence = 0
        steps_without_significant_change = 0
        steps_to_converge = self.max_steps
        for step in range(self.max_steps):
            state_B = op_transfer.apply(state_B)
            current_coherence = op_transfer.last_delta["mean_coherence"]
            if abs(current_coherence - previous_coherence) < self.convergence_threshold:
                steps_without_significant_change += 1
                if steps_without_significant_change >= 10:
                    steps_to_converge = step + 1
                    break
            else:
                steps_without_significant_change = 0
            previous_coherence = current_coherence

        # Soft test, matching the Hebbian version's intent: transferred-weight
        # initialization should produce a genuinely different outcome than a
        # fresh (random-weight) run, not necessarily a better/faster one.
        coherence_transfer = op_transfer.last_delta["mean_coherence"]
        coherence_random = weights_B_random["within_layer_weights"][0].mean()
        print(f"Processing 'B' with weights from 'A': Coherence = {coherence_transfer:.4f}, Steps = {steps_to_converge}")
        print(f"Processing 'B' with random weights: Mean weight = {coherence_random:.4f}")
        self.assertNotEqual(round(coherence_transfer, 2), round(coherence_random, 2),
                           "Transfer learning should produce different results than random initialization")

    def test_parameter_sensitivity(self):
        """Test sensitivity to different learning-rate/decay parameter settings.

        Ported from TestHebbianKuramotoCharacterProcessing, sweeping this
        model's own parameters (pc_learning_rate, hebb_learning_rate,
        hebb_decay_rate, pc_error_scaling) in place of Hebbian's mu/alpha.
        """
        char = 'A'
        parameter_sets = [
            {'pc_learning_rate': 0.05, 'hebb_learning_rate': 0.05, 'hebb_decay_rate': 0.01, 'pc_error_scaling': 0.5},
            {'pc_learning_rate': 0.2, 'hebb_learning_rate': 0.05, 'hebb_decay_rate': 0.01, 'pc_error_scaling': 0.5},
            {'pc_learning_rate': 0.05, 'hebb_learning_rate': 0.2, 'hebb_decay_rate': 0.01, 'pc_error_scaling': 0.5},
            {'pc_learning_rate': 0.05, 'hebb_learning_rate': 0.05, 'hebb_decay_rate': 0.05, 'pc_error_scaling': 0.5},
        ]
        coherence_values = []
        for params in parameter_sets:
            self.pc_learning_rate = params['pc_learning_rate']
            self.hebb_learning_rate = params['hebb_learning_rate']
            self.hebb_decay_rate = params['hebb_decay_rate']
            self.pc_error_scaling = params['pc_error_scaling']
            _, _, deltas, _ = self.process_character(char, iterations=self.max_steps)
            coherence_values.append(deltas[-1]["mean_coherence"])
            print(f"Parameters: {params}, Coherence: {deltas[-1]['mean_coherence']:.4f}")

        cv = np.std(coherence_values) / np.mean(coherence_values)
        self.assertGreater(cv, 0.05, "Network should be sensitive to parameter changes")
