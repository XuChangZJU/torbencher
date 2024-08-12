import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.sub)
class TorchTensorSubTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_sub_correctness(self):
        """
        Test the correctness of torch.Tensor.sub with small scale random parameters.
        """
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        tensor1 = torch.randn(input_size)
        tensor2 = torch.randn(input_size)  # The size of tensor2 should be same as tensor1
        result = tensor1.sub(tensor2)
        return result
