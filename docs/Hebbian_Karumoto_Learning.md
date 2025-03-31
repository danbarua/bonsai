# Implementation Plan: Character Input Test for Hebbian Kuramoto Network

This document outlines a plan for creating a test file that presents 8x12 pixel characters to a Single Layer Kuramoto network and captures the network state after convergence or reaching a maximum number of steps.

## 1. Character Representation

We'll represent characters as 8x12 binary matrices where:
- 1 represents an active pixel (part of the character)
- 0 represents an inactive pixel (background)

We'll create a utility function to define common characters (letters, numbers, symbols) as these binary matrices.

```python
def get_character_matrix(char):
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
        ]).T  # Transpose to get 8x12
        # Add more characters as needed
    }
    return chars.get(char, np.zeros((8, 12)))
```

## 2. Input Mapping to Oscillator State

We'll map the binary character matrix to oscillator phases and perturbations:

- **Phases**: Initialize with random values or specific patterns
- **Frequencies**: Set to uniform values or structured patterns
- **Perturbations**: Map the character matrix to perturbations, where:
  - Active pixels (1) receive a positive perturbation
  - Inactive pixels (0) receive zero or negative perturbation

```python
def create_character_state(char, perturbation_strength=1.0):
    """Create a LayeredOscillatorState from a character."""
    char_matrix = get_character_matrix(char)
    
    # Initialize phases randomly or with a specific pattern
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
```

## 3. Network Processing and Convergence Detection

We'll run the Hebbian Kuramoto operator on the state until either:
- The network reaches a stable state (convergence)
- A maximum number of steps is reached

To detect convergence, we'll track changes in phase coherence or weight structure.

```python
def process_character(char, max_steps=1000, convergence_threshold=1e-4):
    """Process a character through the Hebbian Kuramoto network until convergence or max steps."""
    # Create initial state from character
    state = create_character_state(char)
    
    # Initialize the operator
    op = HebbianKuramotoOperator(dt=0.01, mu=0.1, alpha=0.01)
    
    # Track metrics for convergence detection
    previous_coherence = 0
    steps_without_significant_change = 0
    
    # Process until convergence or max steps
    for step in range(max_steps):
        # Apply the operator
        state = op.apply(state)
        
        # Get current coherence
        current_coherence = op.last_delta["mean_coherence"]
        
        # Check for convergence
        if abs(current_coherence - previous_coherence) < convergence_threshold:
            steps_without_significant_change += 1
            if steps_without_significant_change >= 10:  # Require stability for 10 consecutive steps
                print(f"Converged after {step+1} steps")
                break
        else:
            steps_without_significant_change = 0
        
        previous_coherence = current_coherence
    
    # Return final state and metrics
    return state, op.weights, op.last_delta
```

## 4. State Analysis and Visualization

We'll analyze the final state to understand how the network has processed the character:

```python
def analyze_character_state(state, weights, delta, char):
    """Analyze the final state after processing a character."""
    # Extract phases and reshape to character dimensions
    phases = state.phases[0]
    
    # Calculate phase coherence map (local coherence around each oscillator)
    coherence_map = np.zeros(phases.shape)
    for i in range(phases.shape[0]):
        for j in range(phases.shape[1]):
            # Define a neighborhood around oscillator (i,j)
            i_min, i_max = max(0, i-1), min(phases.shape[0], i+2)
            j_min, j_max = max(0, j-1), min(phases.shape[1], j+2)
            
            # Calculate local coherence
            neighborhood = phases[i_min:i_max, j_min:j_max]
            z = np.exp(1j * neighborhood.flatten())
            coherence_map[i, j] = np.abs(np.mean(z))
    
    # Visualize the results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original character
    axes[0, 0].imshow(get_character_matrix(char), cmap='binary')
    axes[0, 0].set_title(f"Original Character: '{char}'")
    
    # Phase map
    phase_img = axes[0, 1].imshow(phases, cmap='hsv')
    axes[0, 1].set_title("Final Phase Distribution")
    plt.colorbar(phase_img, ax=axes[0, 1])
    
    # Coherence map
    coh_img = axes[1, 0].imshow(coherence_map, cmap='viridis')
    axes[1, 0].set_title("Local Phase Coherence")
    plt.colorbar(coh_img, ax=axes[1, 0])
    
    # Weight matrix (flattened)
    weight_img = axes[1, 1].imshow(weights[0], cmap='coolwarm')
    axes[1, 1].set_title("Final Weight Matrix")
    plt.colorbar(weight_img, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig(f"character_{char}_analysis.png")
    plt.close()
    
    return coherence_map
```

## 5. Test Suite Structure

We'll create a test class that:
1. Tests processing of individual characters
2. Tests the network's ability to distinguish between different characters
3. Tests the stability of the representation across multiple runs

```python
class TestHebbianKuramotoCharacterProcessing(unittest.TestCase):
    def test_single_character_processing(self):
        """Test processing of a single character."""
        char = 'A'
        state, weights, delta = process_character(char)
        coherence_map = analyze_character_state(state, weights, delta, char)
        
        # Assert that coherence is higher in character regions
        char_matrix = get_character_matrix(char)
        avg_coherence_char = np.mean(coherence_map[char_matrix > 0])
        avg_coherence_bg = np.mean(coherence_map[char_matrix == 0])
        self.assertGreater(avg_coherence_char, avg_coherence_bg)
    
    def test_character_distinction(self):
        """Test that different characters produce distinct network states."""
        chars = ['A', 'B', 'C']
        states = []
        
        for char in chars:
            state, _, _ = process_character(char)
            states.append(state.phases[0])
        
        # Calculate pairwise distances between final states
        for i in range(len(chars)):
            for j in range(i+1, len(chars)):
                # Phase distance metric (circular)
                phase_diff = np.abs(np.angle(np.exp(1j * (states[i] - states[j]))))
                mean_diff = np.mean(phase_diff)
                
                # Assert that different characters produce distinct states
                self.assertGreater(mean_diff, 0.1)
    
    def test_processing_stability(self):
        """Test stability of character processing across multiple runs."""
        char = 'A'
        coherence_values = []
        
        # Run multiple times with different random initializations
        for _ in range(5):
            _, _, delta = process_character(char)
            coherence_values.append(delta["mean_coherence"])
        
        # Calculate coefficient of variation (std/mean)
        cv = np.std(coherence_values) / np.mean(coherence_values)
        
        # Assert that results are reasonably stable
        self.assertLess(cv, 0.2)
```

## 6. Additional Features (Optional)

If time permits, we could extend the implementation with:

1. **Character Sequence Processing**: Process sequences of characters and analyze transitions
2. **Noise Robustness**: Test how well the network handles noisy character inputs
3. **Learning Transfer**: Test if learning one character helps with processing similar characters
4. **Parameter Optimization**: Find optimal parameters for character recognition
5. **Comparison with Standard Kuramoto**: Compare with non-Hebbian Kuramoto for this task

## Implementation Steps

1. Create a new file `tests/test_character_processing.py`
2. Implement the character representation functions
3. Implement the state creation and processing functions
4. Implement the analysis and visualization functions
5. Implement the test cases
6. Run and debug the tests
7. Analyze the results and optimize parameters if needed
