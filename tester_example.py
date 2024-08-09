from src.singleTester import SingleTester

tester = SingleTester()

from src.testcase.torch.addmm import TorchAddmmTestCase

tester.run(TorchAddmmTestCase, device="cuda", seed=123)

from src.testcase.torch.quantized_batch_norm import TorchQuantizedUbatchUnormTestCase

tester.run(TorchQuantizedUbatchUnormTestCase, device="cuda", seed=123)
