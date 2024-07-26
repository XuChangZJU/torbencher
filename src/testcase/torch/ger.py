import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.ger)
class TorchGerTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_ger_correctness(self):
        # Generate random dimensions for the input tensors
        dim1 = random.randint(1, 5)
        dim2 = random.randint(1, 5)

        # Create random input tensors 
        input = torch.randn([dim1])  # input is a vector
        vec2 = torch.randn([dim2])  # vec2 is a vector
        result = torch.ger(input, vec2)
        return result
