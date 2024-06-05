
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.from_dlpack)
class TorchFromDlpackTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_from_dlpack_correctness(self):
        import torch.utils.dlpack as dlpack
        tensor = torch.randn(random.randint(1, 10))
        dlpack_tensor = dlpack.to_dlpack(tensor)
        result = torch.from_dlpack(dlpack_tensor)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_from_dlpack_large_scale(self):
        import torch.utils.dlpack as dlpack
        tensor = torch.randn(random.randint(1000, 10000))
        dlpack_tensor = dlpack.to_dlpack(tensor)
        result = torch.from_dlpack(dlpack_tensor)
        return result

