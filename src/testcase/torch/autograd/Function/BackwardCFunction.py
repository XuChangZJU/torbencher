import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.autograd.function.BackwardCFunction)
class TorchAutogradFunctionBackwardcfunctionTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_backward_c_function_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]
    
        # Random tensor for input
        input_tensor = torch.randn(input_size, requires_grad=True)
        # Random tensor for gradient
        grad_output = torch.randn(input_size)
    
        # Perform a simple operation to create a computational graph
        output_tensor = input_tensor * 2
    
        # Perform backward pass
        output_tensor.backward(grad_output)
    
        # Return the gradient of the input tensor
        return input_tensor.grad
    