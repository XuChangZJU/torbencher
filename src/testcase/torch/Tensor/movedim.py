import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.movedim)
class TorchTensorMovedimTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_movedim_correctness(self):
        # Generate random dimension for the tensor
        dim = random.randint(2, 4)
        # Generate random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Generate input size
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate random tensor
        input_tensor = torch.randn(input_size)
        # Generate random source
        source = random.randint(0, dim - 1)
        # Generate random destination, make sure source != destination
        destination = random.randint(0, dim - 2)
        if destination >= source:
            destination += 1
        # Apply movedim
        result = input_tensor.movedim(source, destination)
        return result
