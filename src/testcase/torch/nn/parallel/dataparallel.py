import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.parallel.dataparallel)
class TorchNnParallelDataparallelTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_data_parallel_correctness(self):
        # Randomly generate the number of dimensions for the input tensor
        dim = random.randint(1, 4)
        # Randomly generate the number of elements in each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]
    
        # Create a random input tensor
        input_tensor = torch.randn(input_size)
    
        # Define a simple module to be evaluated in parallel
        class SimpleModule(torch.nn.Module):
            def forward(self, x):
                return x * 2
    