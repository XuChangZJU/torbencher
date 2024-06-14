import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.randint)
class TorchRandintTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_randint_correctness(self):
    # Generate random parameters for randint
    low = random.randint(-10, 10)  # Random low value between -10 and 10
    high = low + random.randint(1, 10)  # Ensure high is greater than low
    dim = random.randint(1, 4)  # Random dimension for the tensor
    num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
    size = [num_of_elements_each_dim for i in range(dim)]

    # Call randint and return the result
