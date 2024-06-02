
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.device_count)
class TorchCudaDeviceCountTestCase(TorBencherTestCaseBase):
    def test_device_count(self, input=None):
        if input is not None:
            result = torch.cuda.device_count()
            return [result, input]
        result = torch.cuda.device_count()
        return [result, None]

