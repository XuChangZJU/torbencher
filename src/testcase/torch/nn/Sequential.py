import torch
import random
from collections import OrderedDict


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.Sequential)
class TorchNnSequentialTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_sequential_correctness(self):
    # Randomly generate dimensions for input tensor
    batch_size = random.randint(1, 4)
    channels = random.randint(1, 3)
    height = random.randint(5, 10)
    width = random.randint(5, 10)
    
    # Create a random input tensor with the generated dimensions
    input_tensor = torch.randn(batch_size, channels, height, width)
    
    # Define a Sequential model with random Conv2d and ReLU layers
    model = torch.nn.Sequential(
        torch.nn.Conv2d(channels, random.randint(1, 10), kernel_size=random.randint(1, 3)),
        torch.nn.ReLU(),
        torch.nn.Conv2d(random.randint(1, 10), random.randint(1, 20), kernel_size=random.randint(1, 3)),
        torch.nn.ReLU()
    )
    
    # Pass the input tensor through the Sequential model
    output_tensor = model(input_tensor)
    return output_tensor
