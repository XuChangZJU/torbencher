
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.get_device_properties)
class TorchCudaGetDevicePropertiesTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_get_device_properties_0(self):
        a = 0
        result = torch.cuda.get_device_properties(a)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_get_device_properties_1(self):
        a = 0
        result = torch.cuda.get_device_properties(device=a)
        return result


