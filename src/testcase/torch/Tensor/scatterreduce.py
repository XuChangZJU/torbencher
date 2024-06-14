import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.scatterreduce)
class TorchTensorScatterreduceTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_scatter_reduce_correctness(self):
        dim = random.randint(0, 3)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim + 1)]
        index_size = input_size.copy()
        index_size[dim] = random.randint(1, input_size[dim])  # index.size(d) <= src.size(d)
        src_size = index_size
        input = torch.randn(input_size)
        index = torch.randint(0, input_size[dim], index_size)
        src = torch.randn(src_size)
        reduce = random.choice(["sum", "prod", "mean", "amax", "amin"])  # Randomly select reduce operation
        result = input.scatter_reduce_(dim, index, src, reduce)
        return result
    