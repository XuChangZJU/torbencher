import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.GroupNorm)
class TorchNnGroupnormTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_groupnorm_correctness(self):
    # Random input size
    dim = random.randint(1, 4)
    num_of_elements_each_dim = random.randint(1, 5)
    input_size = [num_of_elements_each_dim for i in range(dim)]

    # Random number of channels
    num_channels = random.randint(1, 10)

    # Random number of groups, ensuring it divides num_channels
    num_groups = random.choice([i for i in range(1, num_channels + 1) if num_channels % i == 0])

    # Generate random input tensor
    input_tensor = torch.randn([20, num_channels] + input_size)

    # Create GroupNorm instance
    group_norm = torch.nn.GroupNorm(num_groups, num_channels)

    # Apply GroupNorm
    result = group_norm(input_tensor)
    
    return result
