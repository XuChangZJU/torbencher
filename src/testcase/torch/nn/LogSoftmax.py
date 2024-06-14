import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.LogSoftmax)
class TorchNnLogsoftmaxTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_log_softmax_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Random input size
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Randomly generated input tensor
        input_tensor = torch.randn(input_size)
        # Randomly generated dim, which should be less than the dimension of input tensor
        dim = random.randint(0, len(input_size) - 1)
        # Calculate LogSoftmax
        logsoftmax = torch.nn.LogSoftmax(dim=dim)
        # Get the result of LogSoftmax
        result = logsoftmax(input_tensor)
        return result
    
    
    
    