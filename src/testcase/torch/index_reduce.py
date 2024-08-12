import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.index_reduce)
class TorchIndexUreduceTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_index_reduce_correctness(self):
        dim = random.randint(0, 1)  # Random dimension for the reduction, limited to ensure valid indexing
        num_of_elements_each_dim = random.randint(2, 5)  # Random number of elements in each dimension
        input_size = [num_of_elements_each_dim for _ in range(2)]  # Generate random input size for 2D tensor

        input_tensor = torch.randn(input_size)  # Random input tensor
        index_size = input_size[dim]  # Ensure index tensor size matches the dimension being reduced
        # index = torch.randint(0, input_size[dim], (input_size[1 - dim],))  # Random index tensor within bounds of input tensor along dimension 'dim'
        # source = torch.randn(index.size())  # Random source tensor matching the size of index tensor

        if dim == 0:
            # dim=0 的情况，index长度可以随机但是要和source行数匹配；source的列数要和input的列数匹配
            index_length = random.randint(2, 5)
            index = torch.randint(0, input_size[dim], (index_length,))
            source_cols = input_size[1]
            source = torch.randn(index_length, source_cols)
        else:
            # dim=1 的情况
            index_length = random.randint(2, 5)
            index = torch.randint(0, input_size[dim], (index_length,))
            source_rows = input_size[0]
            source = torch.randn(source_rows, index_length)

        reduce = random.choice(['prod', 'mean', 'amax', 'amin'])  # Random reduction operation

        result = torch.index_reduce(input_tensor, dim, index, source, reduce)
        return result
