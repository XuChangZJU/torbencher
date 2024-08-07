import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.scatter_)
class TorchTensorScatterUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_scatter__correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Random input size
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Random self tensor
        self_tensor = torch.randn(input_size)
        # Random dim value
        dim = random.randint(0, len(input_size) - 1)
        # Random index tensor, index.size(d) <= self.size(d) for all dimensions d != dim
        index_size = input_size.copy()
        index_size[dim] = random.randint(1, input_size[dim])
        index_tensor = torch.randint(0, self_tensor.size(dim), index_size)
        # Random src tensor, index.size(d) <= src.size(d) for all dimensions d
        src_size = index_size
        src_tensor = torch.randn(src_size)
        # Call scatter_ function
        result = self_tensor.scatter_(dim, index_tensor, src_tensor)
        return result
