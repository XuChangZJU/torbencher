import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cond)
class TorchCondTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cond_correctness(self):
    # Define two functions that will be used in the conditional
    def true_fn(x):
        return x.cos()
