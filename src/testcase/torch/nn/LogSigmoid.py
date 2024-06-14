import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.LogSigmoid)
class TorchNnLogsigmoidTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_logsigmoid_correctness(self):
    # Randomly generate input size
    dim = random.randint(1, 4)
    num_of_elements_each_dim = random.randint(1, 5)
    input_size = [num_of_elements_each_dim for i in range(dim)]

    # Generate random input tensor
    input_tensor = torch.randn(input_size)

    # Instantiate LogSigmoid module
    logsigmoid_module = torch.nn.LogSigmoid()

    # Calculate LogSigmoid output
    output_tensor = logsigmoid_module(input_tensor)

    return output_tensor
