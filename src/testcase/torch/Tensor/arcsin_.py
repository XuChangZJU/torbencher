import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.arcsin_)
class TorchTensorArcsinUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_arcsin__correctness(self):
        # Randomly generate the dimension of the input tensor.
        dim = random.randint(1, 4)
        # Randomly generate the number of elements in each dimension.
        num_of_elements_each_dim = random.randint(1, 5)
        # Create a list of input sizes.
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate a random tensor.
        input_tensor = torch.randn(input_size) * random.uniform(-1,
                                                                1)  # Make sure the elements are within the valid range of arcsin.
        # Perform the in-place arcsin operation.
        input_tensor.arcsin_()
        return input_tensor
