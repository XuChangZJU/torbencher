import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.special.i0e)
class TorchSpecialI0eTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_i0e_correctness(self):
        # Generate random dimension and size for the input tensor
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Generate random input tensor
        input_tensor = torch.randn(input_size)
    
        # Calculate i0e
        result = torch.special.i0e(input_tensor)
        return result
    