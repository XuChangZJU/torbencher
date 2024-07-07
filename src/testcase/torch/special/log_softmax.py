import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.special.log_softmax)
class TorchSpecialLogsoftmaxTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_log_softmax_correctness(self):
        # Randomly generate the dimension of the input tensor
        dim = random.randint(1, 4)
        # Randomly generate the number of elements for each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Create the input size list
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate a random tensor with the specified input size
        input_tensor = torch.randn(input_size)
        # Randomly select a dimension along which to compute the log_softmax
        dim_to_compute = random.randint(0, len(input_size) - 1)
        # Calculate the log_softmax using the function
        result = torch.special.log_softmax(input_tensor, dim_to_compute)
        return result
    
    
    
    