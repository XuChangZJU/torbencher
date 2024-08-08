import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.index_add)
class TorchIndexUaddTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_index_add_correctness(self):
        dim = random.randint(0, 3)  # Random dimension for index_add
        input_size = [random.randint(1, 5) for _ in range(4)]  # Random size for the input tensor
        index_size = input_size[dim]  # Size of the index tensor should match the chosen dimension of the input tensor
        source_size = input_size.copy()  # Source tensor should have the same size as input except for the indexed dimension
        source_size[
            dim] = index_size  # The indexed dimension of the source tensor should match the size of the index tensor

        input_tensor = torch.randn(input_size)
        index_tensor = torch.randint(0, input_size[dim], (index_size,),
                                     dtype=torch.int64)  # Index values should be within the bounds of the chosen dimension
        source_tensor = torch.randn(source_size)
        result = torch.index_add(input_tensor, dim, index_tensor, source_tensor)
        return result
