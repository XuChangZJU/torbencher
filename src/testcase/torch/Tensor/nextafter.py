import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.nextafter)
class TorchTensorNextafterTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_nextafter_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Random input size
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random tensor1
        tensor1 = torch.randn(input_size)
        # Generate random tensor2
        tensor2 = torch.randn(input_size)

        result = tensor1.nextafter(tensor2)
        return result
