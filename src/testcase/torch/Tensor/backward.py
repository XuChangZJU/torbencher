import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.backward)
class TorchTensorBackwardTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_backward_correctness(self):
        # Generate random dimension and number of elements for the tensor
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Create a random tensor with requires_grad=True
        tensor = torch.randn(input_size, requires_grad=True)
    
        # Compute a result from the tensor to ensure it has a graph
        result = (tensor * 2).sum()
    
        # Call backward() to compute gradients
        result.backward()
    
        # Return the gradient of the tensor
        return tensor.grad
    