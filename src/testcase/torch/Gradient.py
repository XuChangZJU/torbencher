import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.gradient)
class TorchGradientTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_gradient_correctness(self):
    # Random dimension for the tensor (choose between 1 to 4)
    dim = random.randint(1, 4)
    # Random number of elements in each dimension (choose between 1 to 5)
    num_elements = random.randint(1, 5)
    # Create input size list based on the number of dimensions
    input_size = [num_elements for _ in range(dim)]
    
    # Generate a random tensor with the calculated size
    input_tensor = torch.randn(input_size)
    
    # Calculate the gradient of the input tensor
    result = torch.gradient(input_tensor)
    
    return result
