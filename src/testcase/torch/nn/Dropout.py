import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.Dropout)
class TorchNnDropoutTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_dropout_correctness(self):
    # Randomly generate input size
    dim = random.randint(1, 4)
    num_of_elements_each_dim = random.randint(1, 5)
    input_size = [num_of_elements_each_dim for i in range(dim)]

    # Generate random input tensor
    input_tensor = torch.randn(input_size)

    # Generate random p value between 0 and 1
    p = random.uniform(0, 1)

    # Define dropout module
    dropout = torch.nn.Dropout(p)

    # Apply dropout
    output_tensor = dropout(input_tensor)

    return output_tensor
