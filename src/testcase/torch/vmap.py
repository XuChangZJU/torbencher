import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.vmap)
class TorchVmapTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_vmap_correctness(self):
        # Function to be vectorized
        def simple_function(tensor1, tensor2):
            return tensor1 + tensor2
    