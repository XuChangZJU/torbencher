import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.is_shared)
class TorchTensorIssharedTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_is_shared_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        # Create a random CPU tensor
        cpu_tensor = torch.randn(input_size)
        cpu_shared = cpu_tensor.is_shared()

        # Create a random CUDA tensor if CUDA is available
        if torch.cuda.is_available():
            cuda_tensor = torch.randn(input_size, device='cuda')
            cuda_shared = cuda_tensor.is_shared()
        else:
            cuda_shared = None  # CUDA not available

        return cpu_shared, cuda_shared
