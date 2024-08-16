import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.bitwise_and)
class TorchTensorBitwiseUandTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_bitwise_and_correctness(self):
        # Randomly generate dimension and size for input tensors
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random tensors with integer values
        tensor1 = torch.randint(1, 10, input_size)
        tensor2 = torch.randint(1, 10, input_size)
        result = tensor1.bitwise_and(tensor2)
        return result
