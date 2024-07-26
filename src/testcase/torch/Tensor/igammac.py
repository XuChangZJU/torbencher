import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.igammac)
class TorchTensorIgammacTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_igammac_correctness(self):
        # Randomly generate input tensor size
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Generate random input tensors
        input_tensor = torch.rand(input_size)  # input should be in range (0, inf)
        other_tensor = torch.rand(input_size)  # other should be in range (0, inf)
    
        # Calculate igammac
        result = input_tensor.igammac(other_tensor)
        return result
    
    
    
    