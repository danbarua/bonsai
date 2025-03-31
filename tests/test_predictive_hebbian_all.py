import unittest
from tests.test_predictive_hebbian_basic import TestPredictiveHebbianBasic
from tests.test_predictive_hebbian_learning import TestPredictiveHebbianLearning
from tests.test_predictive_hebbian_edge_cases import TestPredictiveHebbianEdgeCases

def load_tests(loader, standard_tests, pattern):
    """Load all tests from the three test modules."""
    suite = unittest.TestSuite()
    
    # Add tests from each test class
    suite.addTests(loader.loadTestsFromTestCase(TestPredictiveHebbianBasic))
    suite.addTests(loader.loadTestsFromTestCase(TestPredictiveHebbianLearning))
    suite.addTests(loader.loadTestsFromTestCase(TestPredictiveHebbianEdgeCases))
    
    return suite

if __name__ == '__main__':
    unittest.main()
