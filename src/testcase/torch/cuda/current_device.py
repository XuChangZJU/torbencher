
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.current_device)
class TorchCudaCurrentDeviceTestCase(TorBencherTestCaseBase):
    def test_current_device(self, input=None):
        if input is not None:
            result = torch.cuda.current_device()
            return [result, input]
        result = torch.cuda.current_device()
        return [result, None]

