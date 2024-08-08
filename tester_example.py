from src.singleTester import SingleTester

tester = SingleTester()

from src.testcase.torch.addmm import TorchAddmmTestCase

tester.run(TorchAddmmTestCase, device="cuda", seed=123)
