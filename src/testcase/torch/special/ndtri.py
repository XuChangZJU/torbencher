import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.special.ndtri)
class TorchSpecialNdtriTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_ndtri_correctness(self):
        # Generate random input tensor
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        input_tensor = torch.rand(input_size)  # Generate random values between 0 and 1
    
        # Calculate ndtri
        result = torch.special.ndtri(input_tensor)
        return result
    