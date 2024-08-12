import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.is_tensor)
class TorchIsUtensorTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_is_tensor_correctness(self):
        # Generate a random tensor
        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]
        tensor = torch.randn(input_size)

        # Test if the generated tensor is indeed a tensor
        result_tensor = torch.is_tensor(tensor)

        # Test if a non-tensor object is recognized as non-tensor
        non_tensor_obj = [1, 2, 3]  # Example non-tensor object
        result_non_tensor = torch.is_tensor(non_tensor_obj)

        return result_tensor, result_non_tensor
