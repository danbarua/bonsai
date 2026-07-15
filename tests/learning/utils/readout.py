"""
Windowed-average readout + nearest-centroid classifier.

Neither model settles to a usable single "final state" in general (Predictive
Hebbian provably doesn't, per the limit-cycle analysis; Hebbian does but we
want a comparison that doesn't depend on one side happening to converge).
So: run each model for a fixed iteration budget T, average the complex phase
representation exp(i*theta) over the last `window` iterations of the INPUT
layer (layer 0), and use that windowed-mean vector as a fixed-length
descriptor for classification.

window should be >= the dominant oscillation period (~123 iterations
measured for Predictive Hebbian on character 'A') so the average actually
smooths a full cycle rather than freezing an arbitrary phase of it.
"""

import numpy as np
from numpy.typing import NDArray

from tests.learning.utils.character_utils import (
    get_character_matrix,
    add_noise_to_character,
    create_occluded_character,
    create_hierarchical_state,
)


def run_and_extract_feature(operator, state, T: int, window: int) -> NDArray[np.float64]:
    """Run `operator` on `state` for T iterations, return a real-valued
    feature vector: the windowed circular mean of the input layer's phases
    over the last `window` iterations, as concatenated [real, imag] parts.
    """
    accum = None
    start_accum = max(0, T - window)
    for i in range(T):
        state = operator.apply(state)
        if i >= start_accum:
            z = np.exp(1j * state.phases[0].flatten())
            accum = z.copy() if accum is None else accum + z
    n_accum = T - start_accum
    mean_z = accum / n_accum
    return np.concatenate([mean_z.real, mean_z.imag])


def make_operator(model_type: str):
    """Fresh operator instance with the benchmark's standard parameters."""
    if model_type == "hebbian":
        from models.hebbian import HebbianKuramotoOperator
        return HebbianKuramotoOperator(dt=0.01, mu=0.05, alpha=0.1)
    elif model_type == "predictive":
        from models.predictive import PredictiveHebbianOperator
        return PredictiveHebbianOperator(
            dt=0.01, pc_learning_rate=0.05, hebb_learning_rate=0.05,
            pc_error_scaling=0.5, pc_precision=1.0, hebb_decay_rate=0.1
        )
    else:
        raise ValueError(model_type)


def build_char_matrix(char: str, noise_level: float = 0.0,
                       occlusion_level: float = 0.0, occlusion_type: str = 'random'):
    char_matrix = get_character_matrix(char)
    if noise_level > 0:
        char_matrix = add_noise_to_character(char_matrix, noise_level)
    if occlusion_level > 0:
        char_matrix = create_occluded_character(char, occlusion_type, occlusion_level)
    return char_matrix


def extract_feature_for_char(model_type: str, char: str, T: int, window: int,
                              noise_level: float = 0.0, occlusion_level: float = 0.0,
                              occlusion_type: str = 'random', seed: int = None) -> NDArray[np.float64]:
    if seed is not None:
        np.random.seed(seed)
    char_matrix = build_char_matrix(char, noise_level, occlusion_level, occlusion_type)
    state = create_hierarchical_state(char_matrix, perturbation_strength=2.0)
    operator = make_operator(model_type)
    return run_and_extract_feature(operator, state, T, window)


def build_templates(model_type: str, characters, T: int, window: int,
                     n_trials: int = 3, base_seed: int = 1000) -> dict:
    """Centroid template per character, averaged over n_trials clean runs
    (different random init each trial) to reduce sensitivity to init noise."""
    templates = {}
    for char in characters:
        feats = []
        for t in range(n_trials):
            f = extract_feature_for_char(model_type, char, T, window,
                                          seed=base_seed + t)
            feats.append(f)
        templates[char] = np.mean(feats, axis=0)
    return templates


def classify(feature: NDArray[np.float64], templates: dict) -> str:
    best_char, best_dist = None, np.inf
    for char, tmpl in templates.items():
        d = np.linalg.norm(feature - tmpl)
        if d < best_dist:
            best_dist = d
            best_char = char
    return best_char


def evaluate_accuracy(model_type: str, characters, templates: dict, T: int, window: int,
                       noise_level: float = 0.0, occlusion_level: float = 0.0,
                       occlusion_type: str = 'random', n_trials: int = 5,
                       base_seed: int = 2000) -> float:
    correct = 0
    total = 0
    for char in characters:
        for t in range(n_trials):
            feat = extract_feature_for_char(
                model_type, char, T, window,
                noise_level=noise_level, occlusion_level=occlusion_level,
                occlusion_type=occlusion_type, seed=base_seed + t
            )
            pred = classify(feat, templates)
            correct += int(pred == char)
            total += 1
    return correct / total
