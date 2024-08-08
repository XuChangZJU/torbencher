import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.take_along_dim)
class TorchTensorTakeUalongUdimTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_take_along_dim_correctness(self):
        dim = random.randint(0, 3)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim + 1)]  # dim+1 to make sure the dimension is valid

        input_tensor = torch.randn(input_size)
        indices_size = input_size.copy()
        indices_size[dim] = random.randint(1, input_size[
            dim])  # The size of dimension 'dim' should be less or equal than the corresponding size in input tensor
        indices = torch.randint(0, input_size[dim], indices_size)
        result = input_tensor.take_along_dim(indices, dim)
        return result
