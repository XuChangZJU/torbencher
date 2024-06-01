import unittest

from src.testcase.torch.add import TorchAddTestCase

if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(TorchAddTestCase("test_add_4d"))
    runner = unittest.TextTestRunner()
    runner.run(suite)
