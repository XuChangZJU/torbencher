import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.Softmax)
class TorchNnSoftmaxTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_softmax_correctness(self):
        # Randomly generate the dimension of the input tensor
        dim = random.randint(1, 4)
        # Randomly generate the number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Generate the input size
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Randomly generate the dimension along which Softmax will be computed
        dim = random.randint(0, len(input_size) - 1)
        # Generate random input tensor
        input_tensor = torch.randn(input_size)
        # Apply Softmax function
        softmax = torch.nn.Softmax(dim=dim)
        result = softmax(input_tensor)
        return result
    
    
    
    