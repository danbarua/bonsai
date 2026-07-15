"""
Classification accuracy test for PredictiveHebbianOperator, using the
windowed-average readout + nearest-centroid classifier built earlier this
project (tests/learning/utils/readout.py).

This is deliberately NOT applied to HebbianKuramotoOperator. Measured
directly (see project changelog): even after fixing the coupling sign bug,
Hebbian scores at-or-near chance (1/7 ~ 0.14-0.20) under this exact readout,
while Predictive Hebbian scores 0.83-0.98 clean. That's a real, unexplained
mismatch between HebbianKuramotoOperator and this particular feature
extraction method (worth its own investigation -- not necessarily a model
bug), and conflating the two models in one test with one threshold would
either be meaningless for Hebbian or misleadingly strict for Predictive.
So: this test covers Predictive Hebbian only, where the readout has been
shown to actually track real, meaningful recognition capability.

Thresholds are set with real margin below values observed across 5
independent trials with different random seeds (n_trials per template/eval
call varies quite a bit run to run -- this model has real trial-to-trial
variance, not just test noise):
    clean:          0.829 - 0.980 observed  -> threshold 0.7
    noise 0.3:      0.343 - 0.657 observed  -> threshold 0.25
    occlusion 0.3:  0.686 - 0.829 observed  -> threshold 0.55
All comfortably above chance (1/7 ~ 0.143) even at the low end.
"""
import unittest

from tests.learning.utils.readout import build_templates, evaluate_accuracy

CHARACTERS = ["A", "B", "C", "1", "2", "+", "-"]
T, WINDOW = 250, 200


class TestPredictiveHebbianClassificationAccuracy(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.templates = build_templates("predictive", CHARACTERS, T, WINDOW, n_trials=3, base_seed=1000)

    def test_clean_accuracy(self):
        acc = evaluate_accuracy("predictive", CHARACTERS, self.templates, T, WINDOW, n_trials=5, base_seed=2000)
        self.assertGreater(acc, 0.7,
            "Predictive Hebbian should reliably classify clean characters well above chance (1/7)")

    def test_noise_robustness(self):
        acc = evaluate_accuracy("predictive", CHARACTERS, self.templates, T, WINDOW,
                                 noise_level=0.3, n_trials=5, base_seed=2000)
        self.assertGreater(acc, 0.25,
            "Predictive Hebbian should retain meaningfully above-chance accuracy at 30% noise")

    def test_occlusion_robustness(self):
        acc = evaluate_accuracy("predictive", CHARACTERS, self.templates, T, WINDOW,
                                 occlusion_level=0.3, n_trials=5, base_seed=2000)
        self.assertGreater(acc, 0.55,
            "Predictive Hebbian should retain good accuracy at 30% occlusion "
            "(previously shown to degrade more gracefully than equivalent noise)")

    def test_occlusion_more_robust_than_noise(self):
        """Specific, previously-observed property: occlusion (clean zeroing of
        missing pixels' sensory error) degrades accuracy less than equivalent-
        level noise (actively wrong sensory error) -- a real, mechanistically
        sensible finding from earlier this project, worth protecting as a
        regression check."""
        noise_acc = evaluate_accuracy("predictive", CHARACTERS, self.templates, T, WINDOW,
                                       noise_level=0.3, n_trials=5, base_seed=2000)
        occlusion_acc = evaluate_accuracy("predictive", CHARACTERS, self.templates, T, WINDOW,
                                           occlusion_level=0.3, n_trials=5, base_seed=2000)
        self.assertGreaterEqual(occlusion_acc, noise_acc,
            "Occlusion should degrade accuracy no more than equivalent-level noise")


if __name__ == "__main__":
    unittest.main()
