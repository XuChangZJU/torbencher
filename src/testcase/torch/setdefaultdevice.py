import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.setdefaultdevice)
class TorchSetdefaultdeviceTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_set_default_device_cuda(self):
    random_device_index = random.randint(0, torch.cuda.device_count() - 1) # Ensure the device index is valid
    random_cuda_device = f'cuda:{random_device_index}'  # Random CUDA device string
    torch.set_default_device(random_cuda_device)  # Set the default device to the random CUDA device
    random_tensor_size = [random.randint(1, 5) for _ in range(random.randint(1, 4))]  # Random tensor size dimensions

    tensor = torch.randn(random_tensor_size)
    return tensor.device
