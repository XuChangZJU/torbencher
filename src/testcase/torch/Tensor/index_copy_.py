import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.index_copy_)
class TorchTensorIndexcopyTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_index_copy_correctness(self):
        dim = random.randint(0, 3)  # Random dimension along which to index
        # Generate random sizes for the tensor
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim + 1)]
        # Create self tensor
        self_tensor = torch.randn(input_size)
        # Generate random indices
        index_size = random.randint(1, input_size[dim])
        index = torch.randint(0, input_size[dim], (index_size,))
        # Create tensor to copy from
        tensor_size = input_size.copy()
        tensor_size[dim] = index_size
        tensor = torch.randn(tensor_size)
        result = self_tensor.index_copy_(dim, index, tensor)
        return result
