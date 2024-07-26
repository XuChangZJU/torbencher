import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.special.gammaincc)
class TorchSpecialGammainccTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_gammaincc_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Generate random input size
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate random input tensor
        input_tensor = torch.rand(input_size) * random.randint(1, 10)  # Scale by a random factor to get different ranges of values
        # Generate random other tensor
        other_tensor = torch.rand(input_size) * random.randint(1, 10)  # Scale by a random factor to get different ranges of values
        # Calculate gammaincc
        result = torch.special.gammaincc(input_tensor, other_tensor)
        return result
    