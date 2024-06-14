import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.lcm)
class TorchTensorLcmTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_lcm_correctness(self):
        # Generate random dimensions for the tensors
        dim = random.randint(1, 4)
        # Generate random number of elements for each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Create a list representing the size of the tensor
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Generate random tensors with integer values
        tensor1 = torch.randint(1, 10, input_size)  # Generate integers between 1 and 10
        tensor2 = torch.randint(1, 10, input_size)  # Generate integers between 1 and 10
        
        # Calculate the least common multiple
        result = tensor1.lcm(tensor2)
        
        return result
    