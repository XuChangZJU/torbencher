from torbencher import SingleTester;
stester = SingleTester();

from src.testcase.torch.addmm import TorchAddmmTestCase;
stester.run(TorchAddmmTestCase, device= "cuda", seed=123);
# from src.testcase.torch.nn.Linear import TorchNnLinearTestCase;
# stester.run(TorchNnLinearTestCase, device= "cuda", seed=123);
