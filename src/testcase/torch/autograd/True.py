import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api
# import torch.autograd.True



# @test_api(torch.autograd) 这里怎么写
class TorchAutogradTrueTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_autograd_true_correctness(self):
        # Random dimension for the tensor
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]
    
        # Random tensor data
        tensor = torch.randn(input_size, requires_grad=True)
        
        # Perform an operation that requires gradient
        result = tensor * 2
        
        # Backward pass to compute gradients
        result.backward(torch.ones_like(tensor))
        
        # Check if gradients are computed correctly
        return tensor.grad
    