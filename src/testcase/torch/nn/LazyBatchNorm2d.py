import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.LazyBatchNorm2d)
class TorchNnLazybatchnorm2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_lazy_batch_norm_2d_correctness(self):
    # Randomly generate dimensions for the input tensor
    batch_size = random.randint(1, 4)  # Random batch size
    num_channels = random.randint(1, 4)  # Random number of channels
    height = random.randint(1, 5)  # Random height of the tensor
    width = random.randint(1, 5)  # Random width of the tensor

    # Create a random input tensor with the generated dimensions
    input_tensor = torch.randn(batch_size, num_channels, height, width)

    # Randomly generate parameters for LazyBatchNorm2d
    eps = random.uniform(1e-6, 1e-4)  # Random epsilon value for numerical stability
    momentum = random.uniform(0.05, 0.15)  # Random momentum value for running mean and variance

    # Initialize LazyBatchNorm2d with the generated parameters
    lazy_batch_norm = torch.nn.LazyBatchNorm2d(eps, momentum)

    # Apply the LazyBatchNorm2d to the input tensor
    result = lazy_batch_norm(input_tensor)
    return result
