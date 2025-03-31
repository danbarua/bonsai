"""
Character utilities for learning tests.

This module provides utilities for creating and manipulating character matrices
for use in oscillator-based learning tests.
"""

import numpy as np
from dynamics.oscillators import LayeredOscillatorState

# Dictionary of predefined character matrices
CHARACTER_MATRICES = {
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
    '+': np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ]).T,
    '-': np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
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

def get_character_matrix(char):
    """
    Return an 8x12 binary matrix representing the given character.
    
    Args:
        char: The character to get the matrix for
        
    Returns:
        An 8x12 numpy array representing the character
    """
    return CHARACTER_MATRICES.get(char, np.zeros((8, 12)))

def add_noise_to_character(char_matrix, noise_level=0.1):
    """
    Add random noise to a character matrix.
    
    Args:
        char_matrix: The character matrix to add noise to
        noise_level: The probability of flipping each pixel
        
    Returns:
        A noisy version of the character matrix
    """
    # Generate random noise (0 or 1) with probability noise_level
    noise = np.random.binomial(1, noise_level, char_matrix.shape)
    # XOR the noise with the character matrix to flip bits
    noisy_matrix = np.logical_xor(char_matrix, noise).astype(float)
    return noisy_matrix

def create_ambiguous_character(char1, char2, ambiguity_level=0.5):
    """
    Create an ambiguous character by blending two characters.
    
    Args:
        char1: The first character
        char2: The second character
        ambiguity_level: The level of ambiguity (0.0 to 1.0)
        
    Returns:
        A blended character matrix
    """
    char1_matrix = get_character_matrix(char1)
    char2_matrix = get_character_matrix(char2)
    
    # Create a mask for blending
    mask = np.random.binomial(1, ambiguity_level, char1_matrix.shape)
    
    # Blend the characters
    ambiguous_matrix = np.where(mask, char2_matrix, char1_matrix)
    return ambiguous_matrix

def create_occluded_character(char, occlusion_type='horizontal', occlusion_level=0.3):
    """
    Create a partially occluded character.
    
    Args:
        char: The character to occlude
        occlusion_type: Type of occlusion ('horizontal', 'vertical', or 'random')
        occlusion_level: The level of occlusion (0.0 to 1.0)
        
    Returns:
        An occluded character matrix
    """
    char_matrix = get_character_matrix(char)
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

def create_single_layer_state(char_matrix, perturbation_strength=1.0):
    """
    Create a single-layer oscillator state from a character matrix.
    
    Args:
        char_matrix: The character matrix to create a state from
        perturbation_strength: The strength of the perturbation
        
    Returns:
        A LayeredOscillatorState with a single layer
    """
    # Initialize phases randomly
    phases = [np.random.uniform(0, 2*np.pi, char_matrix.shape)]
    
    # Set uniform frequencies
    frequencies = [np.ones(char_matrix.shape)]
    
    # Map character to perturbations
    perturbations = [char_matrix * perturbation_strength]
    
    # Create the state
    return LayeredOscillatorState(
        _phases=phases,
        _frequencies=frequencies,
        _perturbations=perturbations,
        _layer_names=["Character Layer"],
        _layer_shapes=[char_matrix.shape]
    )

def create_hierarchical_state(input_matrix, layer_shapes=None, perturbation_strength=1.0):
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
                try:
                    from scipy.ndimage import zoom
                    zoom_factors = (shape[0]/input_matrix.shape[0], shape[1]/input_matrix.shape[1])
                    resized_input = zoom(input_matrix, zoom_factors, order=0)
                except ImportError:
                    # Fallback if scipy is not available
                    resized_input = input_matrix
                    print("Warning: scipy not available. Using original input matrix.")
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

def calculate_local_coherence(phase_data):
    """
    Calculate local phase coherence map.
    
    Args:
        phase_data: The phase data to calculate coherence for
        
    Returns:
        A coherence map of the same shape as phase_data
    """
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
