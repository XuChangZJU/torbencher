import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.nn.init.trunc_normal_)
class TorchNnInitTruncnormalTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_trunc_normal_correctness(self):
        # Random dimension for the tensor
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Random input size
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate a random tensor with the specified size
        tensor = torch.empty(input_size)
        # Generate random mean and std
        mean = random.uniform(-10.0, 10.0)
        std = random.uniform(0.1, 10.0)
        # Generate random a and b, making sure a <= mean <= b
        a = random.uniform(-20.0, mean)
        b = random.uniform(mean, 20.0)
        # Apply trunc_normal_
        result = torch.nn.init.trunc_normal_(tensor, mean, std, a, b)
        return result
    