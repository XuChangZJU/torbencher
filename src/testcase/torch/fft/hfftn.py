import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.fft.hfftn)
class TorchFftHfftnTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_hfftn_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Generate input_size
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Append the last dimension as (2^n + 1) to make sure the last dimension is valid.
        input_size[-1] = 2**(random.randint(1, 10)) + 1
        # Generate random input tensor
        input_tensor = torch.randn(input_size)
        # Generate random s
        s = [random.randint(1, 10) for i in range(dim)]
        # Generate random dim
        dim = [i for i in range(len(input_size))]
        # Calculate hfftn
        result = torch.fft.hfftn(input_tensor, s, dim)
        # Return the calculated result
        return result
    
    
    
    