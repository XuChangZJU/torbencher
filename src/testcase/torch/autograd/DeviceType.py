
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.autograd.DeviceType)
class TorchAutogradDeviceTypeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_devicetype_correctness(self):
        device_type = torch.autograd.DeviceType(random.choice(["cpu", "cuda", "meta", "mkldnn", "opengl", "opencl", "ideep", "hip", "xpu", "vulkan", "mps", "rocm", "onednn", "mkl", "sparse_csr", "sparse_coo", "sparse_bsr", "sparse_csc", "sparse_dia", "sparse_block"]))
        result = device_type
        return result

    @test_api_version.larger_than("1.1.3")
    def test_devicetype_large_scale(self):
        device_type = torch.autograd.DeviceType(random.choice(["cpu", "cuda", "meta", "mkldnn", "opengl", "opencl", "ideep", "hip", "xpu", "vulkan", "mps", "rocm", "onednn", "mkl", "sparse_csr", "sparse_coo", "sparse_bsr", "sparse_csc", "sparse_dia", "sparse_block"]))
        result = device_type
        return result


