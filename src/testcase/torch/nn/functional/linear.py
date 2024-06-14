import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.linear)
class TorchNnFunctionalLinearTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_linear_correctness(self):
    # Define the dimensions of the input tensor
    dim = random.randint(1, 4)
    num_of_elements_each_dim = random.randint(1, 5)
    input_size = [num_of_elements_each_dim for i in range(dim)]

    # Add in_features dimension
    in_features = random.randint(1, 10)
    input_size.append(in_features)

    # Define the output features
    out_features = random.randint(1, 10)

    # Generate random input tensor
    input_tensor = torch.randn(input_size)

    # Generate random weight tensor
    weight_tensor = torch.randn(out_features, in_features)

    # Generate random bias tensor
    bias_tensor = torch.randn(out_features)

    # Calculate the linear transformation
    result = torch.nn.functional.linear(input_tensor, weight_tensor, bias_tensor)
    
    return result
