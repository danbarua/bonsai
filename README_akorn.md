# Artificial Kuramoto Oscillatory Neurons

Here's a breakdown of the "[Artificial Kuramoto Oscillatory Neurons](https://takerum.github.io/akorn_project_page/)" (AKOrN) [paper](https://arxiv.org/abs/2410.13821) in plain English, focusing on the key deviations from typical Kuramoto models and how their vectorized model works, along with potential NumPy implementations:

## In Plain English

- **The Core Idea:** The paper introduces a new type of artificial neuron called an Artificial Kuramoto Oscillatory Neuron (AKOrN). Instead of just turning on or off (like in traditional neural networks), these neurons are like tiny oscillators that can synchronize with each other. This synchronization is based on a mathematical model called the Kuramoto model.
- **Why Oscillators?** The authors argue that using oscillators allows the network to "bind" features together. Imagine recognizing a blue square: some neurons might represent "blue," others "square," and the synchronization of these neurons represents the whole object. Oscillators also naturally create "traveling waves," which are seen in the brain and might be important for memory and other cognitive functions.
- **Key Differences from Traditional Kuramoto:**

  - **Multi-Dimensional Vectors:** Instead of representing each oscillator with a single phase (a number between 0 and 2π), AKOrNs use a multi-dimensional vector that rotates on a sphere. This gives each neuron more representational power.
  - **Learnable Parameters:** The connections between oscillators and their natural frequencies are learned during training, allowing the network to adapt to specific tasks.
  - **Conditional Stimuli:** Each oscillator is also influenced by an external "stimulus" that depends on the input data. This helps the oscillators latch onto relevant features in the input.
  - **Asymmetric Connections:** Unlike some theoretical Kuramoto models, the connections between AKOrNs don't have to be symmetric. This makes the model more biologically realistic and improves performance.

- **What Problems Do They Solve?** The authors show that AKOrNs can improve performance on a variety of tasks, including:

  - **Unsupervised Object Discovery:** Finding objects in images without being told what they are.
  - **Reasoning:** Solving Sudoku puzzles.
  - **Robustness:** Being resistant to noise and adversarial attacks.
  - **Uncertainty Quantification:** Knowing when the network is unsure of its predictions.

**Key Equations and NumPy Translation**

The core of AKOrN is the update rule for the oscillators. Here's the main equation from the paper:

$$\dot{\mathbf{x}}_i = \Omega_i \mathbf{x}_i + \text{Proj}_{\mathbf{x}_i} \left( \mathbf{c}_i + \sum_j J_{ij} \mathbf{x}_j \right)$$

Let's break this down and translate it into NumPy code:

- **$\mathbf{x}_i$:** This is the state of the $i$-th oscillator, represented as a vector on a sphere (unit vector).
- **$\dot{\mathbf{x}}_i$:** This is the time derivative of $\mathbf{x}_i$, i.e., how the oscillator's state changes over time.
- **$\Omega_i$:** This is an anti-symmetric matrix that determines the natural frequency and rotation of the $i$-th oscillator.
- **$\mathbf{c}_i$:** This is the conditional stimulus for the $i$-th oscillator, derived from the input data.
- **$J_{ij}$:** This is the connection strength between the $i$-th and $j$-th oscillators.
- **$\text{Proj}_{\mathbf{x}_i}$:** This is the projection operator that projects a vector onto the tangent space of the sphere at $\mathbf{x}_i$. This ensures that the updated oscillator state remains on the sphere.

--

### Explanation of the Code:

1. **`kuramoto_update(x, omega, c, J, dt)`:** This function takes the current oscillator states (`x`), natural frequencies (`omega`), conditional stimuli (`c`), coupling strengths (`J`), and time step (`dt`) as input.
2. **`natural_frequency = np.einsum('nij,nj->ni', omega, x)`:** This calculates the natural frequency term ($\Omega_i \mathbf{x}_i$) using `np.einsum` for efficient matrix multiplication across all oscillators.
3. **`interaction = c.copy()`:** Initializes the interaction term with the conditional stimuli.
4. **Loop for Interaction:** The code iterates through each oscillator and calculates the influence from all other oscillators based on the coupling strengths (`J`).
5. **`projection = interaction - np.sum(interaction * x, axis=1, keepdims=True) * x`:** This implements the projection operator. It subtracts the component of the `interaction` vector that is parallel to the oscillator's current state (`x`), leaving only the component that is tangent to the sphere.
6. **`x_dot = natural_frequency + projection`:** Calculates the time derivative of the oscillator state.
7. **`x_new = x + dt * x_dot`:** Updates the oscillator state using a simple Euler integration step.
8. **`x_new = x_new / np.linalg.norm(x_new, axis=1, keepdims=True)`:** Normalizes the updated state to ensure it remains on the unit sphere.

---

## Energy-based Voting: Getting a More Confident Answer from AKOrN

Imagine you're trying to solve a Sudoku puzzle. Sometimes, you might try a few different approaches before you find the right one. Energy-based voting is a similar idea for the AKOrN model, especially when it's used for tasks like solving Sudoku.

Here's the breakdown:

1. **The "Energy" Concept:** The AKOrN model has an "energy" value associated with its state. This energy is calculated based on how well the oscillators are synchronized and aligned with the input data (the Sudoku board in this case). The lower the energy, the more stable and consistent the solution is considered to be.

2. **Multiple Attempts:** Instead of just running the AKOrN model once to get a single answer, you run it multiple times. Each time, you start the oscillators from a slightly different random initial state. Think of this as trying different starting points for solving the Sudoku.

3. **Collect the Answers and Their Energies:** Each run of the AKOrN model produces a potential solution to the Sudoku puzzle, along with an energy value.

4. **Vote Based on Energy:** Instead of simply averaging the different solutions (like in a typical "majority voting" scheme), you select the solution that has the *lowest* energy. The idea is that the lowest-energy solution is the most stable and therefore the most likely to be correct.

**Why Does This Work?**

- **AKOrN as an Energy-Based Model:** The authors found that even though they didn't explicitly train the AKOrN model to minimize energy, the energy value still provides a good indication of the solution's correctness.
- **Escaping Bad Local Minima:** By starting from different random initial states, the AKOrN model can explore different parts of the solution space and potentially escape from "bad" local minima (solutions that seem good at first but are actually incorrect).
- **Finding the Most Stable Solution:** The lowest-energy solution represents the most stable and consistent state of the oscillators, which is more likely to correspond to the correct answer.

**In Simple Terms**

Energy-based voting is like asking a group of experts to solve a problem, and then choosing the solution from the expert who seems the most confident (has the lowest "energy").


**How to Implement It (Conceptual NumPy Code)**

```python
import numpy as np

def energy_based_voting(model, input_data, num_samples):
    """
    Applies energy-based voting to the AKOrN model.

    Args:
        model: The AKOrN model.
        input_data: The input data (e.g., Sudoku board).
        num_samples: The number of random initial states to try.

    Returns:
        The solution with the lowest energy.
    """

    solutions = []
    energies = []

    for _ in range(num_samples):
        # 1. Initialize the oscillators with random states
        initial_state = initialize_random_oscillators(model.num_oscillators, model.oscillator_dim)

        # 2. Run the AKOrN model to get a solution and its energy
        solution, energy = model.forward(input_data, initial_state)

        solutions.append(solution)
        energies.append(energy)

    # 3. Find the solution with the lowest energy
    best_solution_index = np.argmin(energies)
    best_solution = solutions[best_solution_index]

    return best_solution
```

**Key Takeaway**

Energy-based voting is a simple but effective technique to improve the performance of AKOrN by leveraging the model's inherent energy landscape. It's particularly useful for tasks where the energy value is a good indicator of solution quality.

## Explanation of made-up ML term "Energy" in Plain English

"Energy" they define is directly related to, and can be interpreted as, a measure of **coherence** or a variant of the **Kuramoto order parameter**. It's *not* a measure of energy in the traditional physics sense (like kinetic or potential energy).

Here's how it maps to well-understood concepts:

- **Kuramoto Order Parameter:** The Kuramoto order parameter (often denoted as *r*) measures the degree of synchronization in a population of oscillators. It's calculated as the magnitude of the average complex phase:

    $r = \left| \frac{1}{N} \sum_{j=1}^{N} e^{i\theta_j} \right|$

    where $\theta_j$ is the phase of the *j*-th oscillator and *N* is the number of oscillators.  When the oscillators are perfectly synchronized, *r* = 1. When they are randomly distributed, *r* is close to 0.

- **AKOrN's Energy Function:** The AKOrN paper defines the energy function as:

    $E = -\frac{1}{2} \sum_{i,j} \mathbf{x}_i^T J_{ij} \mathbf{x}_j - \sum_i \mathbf{c}_i^T \mathbf{x}_i$

    The first term, $-\frac{1}{2} \sum_{i,j} \mathbf{x}_i^T J_{ij} \mathbf{x}_j$, encourages alignment between oscillators that are strongly connected (large $J_{ij}$). When oscillators are aligned, this term becomes more negative, lowering the energy. The second term, $-\sum_i \mathbf{c}_i^T \mathbf{x}_i$, encourages alignment between the oscillators and the conditional stimuli.

- **Coherence as a Proxy for Low Energy:** When the oscillators are coherent (synchronized and aligned with the stimuli), the terms in the AKOrN energy function become more negative, resulting in a lower energy value. Therefore, a lower energy value *implies* a higher degree of coherence.
- **Energy-Based Voting as Coherence Maximization:** By selecting the solution with the lowest energy, the AKOrN model is effectively selecting the solution where the oscillators are most coherent and aligned with the input data.

**Why "Energy" is Used (Even Though It's Not Traditional Energy)**

The authors likely use the term "energy" because:

1. **Analogy to Physical Systems:** It draws an analogy to physical systems where systems tend to minimize their energy to reach stable states.
2. **Lyapunov Function:** The proofs in the appendices show that the defined "energy" acts as a Lyapunov function, which guarantees stability.
3. **Conciseness:** "Energy" is a shorter and more evocative term than "negative coherence" or "stability metric."

**In Summary**

The "energy" in the AKOrN paper is essentially a measure of coherence or a variant of the Kuramoto order parameter. It's used as a heuristic for convergence because a lower "energy" (higher coherence) indicates a more stable and consistent solution. The authors are using the term "energy" in a slightly non-standard way to draw an analogy to physical systems and to leverage the concept of Lyapunov functions for proving stability.

---

## Couplign Asymmetry

Okay, I've reviewed the context. When the AKOrN paper discusses "asymmetry" in the coupling matrix ($J_{ij}$), they are referring to **asymmetry of values**, not dimensions.

Here's why:

- **Dimensions are Implicitly Defined:** The dimensions of the coupling matrix $J_{ij}$ are implicitly determined by the number of oscillators ($N$) and the dimensionality of each oscillator ($N$). If you have $N$ oscillators, the coupling matrix will be an $N \times N$ matrix (or a tensor if each connection has internal weights). The paper doesn't suggest that the number of oscillators would be different for the "sending" vs. "receiving" side of a connection.
- **Symmetry vs. Asymmetry of Values:** The key distinction the paper makes is whether the *values* in the coupling matrix are symmetric. In a symmetric coupling matrix, $J_{ij} = J_{ji}$ (or $J_{ij} = J_{ji}^T$ if the connections are matrices themselves). This means the connection strength from oscillator *i* to oscillator *j* is the same as the connection strength from oscillator *j* to oscillator *i*. In an asymmetric coupling matrix, this is not the case. The connection strengths can be different in each direction.
- **Biological Plausibility:** The paper explicitly mentions that asymmetric connections are more biologically plausible because synapses in the brain are not typically symmetric.
- **Performance Improvement:** The paper states that they found better performance with asymmetric connections compared to symmetric connections, particularly in the Sudoku reasoning task.

**In NumPy Terms**

If `J` is your coupling matrix in NumPy:

- **Symmetric:** `np.allclose(J, J.T)` would evaluate to `True` (or `np.allclose(J[i,j], J[j,i].T)` if the connections are matrices themselves).
- **Asymmetric:** `np.allclose(J, J.T)` would evaluate to `False`.

**Why Asymmetry Matters**

Asymmetry in the coupling matrix allows for more complex and flexible interactions between oscillators. It breaks the constraint that oscillators must influence each other equally, which can be beneficial for learning complex patterns and relationships in the data. In the context of neural networks, asymmetric connections are the norm, allowing for directed information flow and more sophisticated computations.

---

## Incorporating insights from [[AKORN Analysis]]

Okay, I've reviewed the `EnhancedHebbianKuramotoOperator` code and the insights from the AKOrN paper. Here's what I would change, and why:

**1. Multi-Dimensional Oscillator States:**

- **Current Implementation:** The current implementation uses a single phase value for each oscillator.
- **AKOrN Insight:** AKOrN uses multi-dimensional vectors to represent oscillator states, which significantly increases the representational capacity of each oscillator.
- **Proposed Change:** Modify the `EnhancedHebbianKuramotoOperator` to use multi-dimensional vectors for the oscillator states. This would involve:
    - Changing the `state.phases` to be a NumPy array of shape `(num_oscillators, oscillator_dim)`, where `oscillator_dim` is the dimensionality of each oscillator.
    - Modifying the update rules to operate on these multi-dimensional vectors.
    - Normalizing the oscillator states after each update to ensure they remain on the unit sphere.

**2. Learnable Natural Frequencies:**

- **Current Implementation:** The current implementation uses fixed natural frequencies (implicitly determined by `state.frequencies`).
- **AKOrN Insight:** AKOrN learns the natural frequencies of the oscillators, allowing the network to adapt to specific tasks.
- **Proposed Change:** Add a learnable parameter for the natural frequencies. This could be implemented as a NumPy array of shape `(num_oscillators, oscillator_dim, oscillator_dim)` representing the anti-symmetric matrices $\Omega_i$ from the AKOrN paper. You would need to use a deep learning framework (like PyTorch or TensorFlow) to optimize these parameters during training.

**3. Conditional Stimuli:**

- **Current Implementation:** The current implementation does not explicitly incorporate conditional stimuli.
- **AKOrN Insight:** AKOrN uses conditional stimuli ($c_i$) to influence the oscillators based on the input data.
- **Proposed Change:** Add a conditional stimulus term to the update rule. This would involve:
    - Creating a mechanism to generate conditional stimuli based on the input data. This could involve using a convolutional neural network or other feature extraction techniques.
    - Adding the conditional stimulus term to the phase update equation.

**4. Asymmetric Connections:**

- **Current Implementation:** The current implementation derives the coupling strengths from the cosine of the phase differences, which results in a symmetric coupling matrix.
- **AKOrN Insight:** AKOrN performs better with asymmetric connections.
- **Proposed Change:** Introduce asymmetry into the coupling matrix. This could be achieved by:
    - Adding a learnable asymmetry term to the coupling strength calculation.
    - Learning the coupling strengths directly, without imposing symmetry.

**5. Projection Operator:**

- **Current Implementation:** The current implementation does not explicitly use a projection operator to keep the oscillators on the sphere.
- **AKOrN Insight:** The projection operator is crucial for ensuring that the oscillator states remain on the unit sphere in AKOrN.
- **Proposed Change:** Implement the projection operator in the phase update rule. This would involve subtracting the component of the update vector that is parallel to the oscillator's current state.

**6. Energy-Based Voting (for applicable tasks):**

- **Current Implementation:** The current implementation does not use energy-based voting.
- **AKOrN Insight:** Energy-based voting can improve performance by selecting the solution with the lowest energy (highest coherence).
- **Proposed Change:** If you are using the `EnhancedHebbianKuramotoOperator` for tasks where multiple solutions are possible (e.g., Sudoku), implement energy-based voting to select the best solution.

**7. Monitor Coherence (Instead of Just "Stability"):**

- **Current Implementation:** The current implementation has a stub for `_analyze_stability()`.
- **AKOrN Insight:** The "energy" (which is really coherence) is a good indicator of solution quality.
- **Proposed Change:** Instead of just looking at abstract "stability metrics," calculate and log the Kuramoto order parameter (or a similar measure of coherence) in the `last_delta`. This will give you a more direct measure of how well the oscillators are synchronizing.


**Important Notes:**

- **Deep Learning Framework:** Implementing learnable natural frequencies and asymmetric connections will require a deep learning framework like PyTorch or TensorFlow to optimize the parameters during training.
- **Computational Cost:** Increasing the dimensionality of the oscillators will increase the computational cost of the model. You may need to experiment with different values of `oscillator_dim` to find a good balance between performance and efficiency.
- **Stability:** Introducing asymmetry and learnable parameters can make the model more difficult to train and may require careful tuning of hyperparameters to ensure stability.

By incorporating these changes, you can create a Hebbian-Kuramoto model that more closely resembles the AKOrN architecture and potentially achieve better performance on your tasks. Remember to experiment with different parameter settings and optimization techniques to find the best configuration for your specific problem.

---

Key changes and explanations:

*   **Connectome-Inspired Coupling:**
    *   The `initialize_connectome_inspired_coupling` method creates a coupling matrix with short-range dense connections and long-range sparse connections, inspired by the human connectome.
    *   The `add_hub_structure` method enhances connectivity to and from hub regions, further mimicking the brain's organization.
*   **Harmonic Resonance:**
    *   The `detect_harmonic_relationships` method identifies when oscillator frequencies are in simple rational relationships.
    *   The `update_coupling_from_resonance` method strengthens the cross-dimensional coupling between oscillators that are in harmonic resonance.
*   **Self-Organizing Intelligence:**
    *   The `detect_stable_representations` method identifies when the system forms stable phase patterns.
    *   The `extract_pattern_representation` method extracts a representation of the current phase pattern.
    *   The `pattern_distance` method calculates the distance between two pattern representations.
    *   The `store_new_pattern` method stores a new pattern in memory.
    *   The `update_pattern_relationships` method strengthens the connections between patterns that activate in sequence.
    *   The `discover_patterns` method orchestrates the pattern discovery and storage process.
*   **Configuration:**
    *   A `config` dictionary is used to store various parameters that control the behavior of the system. This makes it easy to adjust the system's behavior without modifying the code.
*   **Metrics:**
    *   The `last_delta` dictionary now includes the number of patterns stored in memory, which provides a measure of the system's learning progress.