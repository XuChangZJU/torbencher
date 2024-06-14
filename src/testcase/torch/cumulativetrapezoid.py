import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cumulativetrapezoid)
class TorchCumulativetrapezoidTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cumulative_trapezoid_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(2, 5)  # Random number of elements each dimension, at least 2 to calculate diff
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        y = torch.randn(input_size)
        result = torch.cumulative_trapezoid(y)
        return result
    