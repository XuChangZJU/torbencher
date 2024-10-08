import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.fmod_)
class TorchTensorFmodUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_fmod__correctness(self):
        # Generate random dimension for the tensors
        dim = random.randint(1, 4)
        # Generate random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Generate input_size
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random tensor1
        tensor1 = torch.randn(input_size)
        # Generate random divisor
        divisor = random.uniform(0.1, 10.0)
        # Calculate the result of fmod_
        result = tensor1.fmod_(divisor)
        # Return the result of fmod_
        return result
