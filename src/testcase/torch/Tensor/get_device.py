import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.get_device)
class TorchTensorGetUdeviceTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_get_device_correctness(self):
        # Randomly decide whether to create a CUDA tensor or a CPU tensor
        use_cuda = random.choice([True, False])

        if use_cuda and torch.cuda.is_available():
            # Create a random tensor on a random CUDA device
            device_id = random.randint(0, torch.cuda.device_count() - 1)
            tensor = torch.randn(3, 4, 5, device=f'cuda:{device_id}')
            result = tensor.get_device()
        else:
            # Create a random tensor on the CPU
            tensor = torch.randn(3, 4, 5)
            result = tensor.get_device()

        return result
