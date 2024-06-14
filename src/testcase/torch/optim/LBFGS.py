import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.optim.LBFGS)
class TorchOptimLbfgsTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_LBFGS_correctness(self):
    # Define the function to optimize
    def func(x):
        return torch.norm(x)**2 
