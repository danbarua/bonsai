# Learning Tests

This directory contains tests for evaluating the learning capabilities of different oscillator-based models, with a focus on character recognition tasks.

## Directory Structure

- `hebbian/`: Tests for Hebbian Kuramoto models
- `predictive/`: Tests for Predictive Hebbian models
- `utils/`: Shared utilities for learning tests

## Character Processing Tests

The character processing tests evaluate how different oscillator-based models process and respond to character inputs represented as 8x12 binary matrices. These tests focus on:

1. **Basic Character Processing**: How well the models process individual characters
2. **Character Distinction**: Whether different characters produce distinct network states
3. **Noise Robustness**: How well the models handle noisy character inputs
4. **Ambiguity Resolution**: How the models resolve ambiguous characters (blends of two characters)
5. **Occlusion Handling**: How the models handle partially occluded characters

## Running the Tests

To run all the Hebbian Kuramoto character processing tests:

```bash
python -m unittest tests.learning.hebbian.test_character_processing
```

To run all the Predictive Hebbian character processing tests:

```bash
python -m unittest tests.learning.predictive.test_character_processing
```

To run a specific test:

```bash
python -m unittest tests.learning.hebbian.test_character_processing.TestHebbianKuramotoCharacterProcessing.test_single_character_processing
```

## Visualization

The tests generate visualizations in the `plots/` directory:

- `plots/hebbian/`: Visualizations for Hebbian Kuramoto models
- `plots/predictive/`: Visualizations for Predictive Hebbian models
- `plots/comparison/`: Comparative visualizations between different models

## Key Findings

### Hebbian Kuramoto Model

- Forms coherent phase patterns in response to character inputs
- Shows distinct phase patterns for different characters
- Demonstrates some robustness to noise and occlusion
- Learning (weight adaptation) enhances pattern formation compared to standard Kuramoto

### Predictive Hebbian Model

- Creates hierarchical representations with increasing abstraction in higher layers
- Shows interesting feature extraction capabilities
- Can reconstruct input from higher-layer representations
- Demonstrates different disambiguation properties compared to Hebbian Kuramoto

## Extending the Tests

To add new tests:

1. Create a new test file in the appropriate directory
2. Inherit from `CharacterProcessingBaseTest` in `tests/learning/utils/base_test.py`
3. Implement the required test methods
4. Use the visualization utilities in `tests/learning/utils/viz_utils.py` to visualize results

## Character Utilities

The `character_utils.py` module provides utilities for creating and manipulating character matrices:

- `get_character_matrix(char)`: Get a binary matrix for a character
- `add_noise_to_character(char_matrix, noise_level)`: Add noise to a character
- `create_ambiguous_character(char1, char2, ambiguity_level)`: Create a blend of two characters
- `create_occluded_character(char, occlusion_type, occlusion_level)`: Create a partially occluded character
- `create_single_layer_state(char_matrix)`: Create a single-layer oscillator state
- `create_hierarchical_state(input_matrix)`: Create a hierarchical oscillator state
