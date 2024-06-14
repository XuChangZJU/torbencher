import unittest

# Import the test case
from src.testcase.torch.nn.functional.fold import TorchNnFunctionalFoldTestCase

if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add the test case to the suite
    suite.addTests(loader.loadTestsFromTestCase(TorchNnFunctionalFoldTestCase))

    runner = unittest.TextTestRunner(verbosity=3)
    runner.run(suite)