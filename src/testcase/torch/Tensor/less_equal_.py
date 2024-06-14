import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.less_equal_)
class TorchTensorLessequalTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_less_equal__correctness(self):
        # Randomly generate the dimension of the input tensors
        dim = random.randint(1, 4)
        # Randomly generate the number of elements in each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Create the input size list for the tensors
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Generate random tensors
        input_tensor = torch.randn(input_size)
        other_tensor = torch.randn(input_size)
    
        # Perform the in-place less_equal_ operation
        input_tensor.less_equal_(other_tensor)
    
        # Return the modified input tensor
        return input_tensor
    
    
    
    