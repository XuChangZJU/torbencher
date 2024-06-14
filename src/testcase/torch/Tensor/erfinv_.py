import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.erfinv_)
class TorchTensorErfinvTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_erfinv__correctness(self):
        # Randomly generate the dimension of the input tensor
        dim = random.randint(1, 4)
        # Randomly generate the number of elements in each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Create the input size list for the tensor
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate a random tensor with values between -1 and 1
        input_tensor = torch.rand(input_size) * 2 - 1 
        # Apply the erfinv_ operation in-place
        input_tensor.erfinv_()
        # Return the modified tensor
        return input_tensor
    
    
    
    