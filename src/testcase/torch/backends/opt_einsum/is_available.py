import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.backends.opt_einsum.is_available)
class TorchBackendsOptUeinsumIsUavailableTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_opt_einsum_is_available(self):
        # Check if opt_einsum is available
        is_available = torch._C._has_opt_einsum()
        return is_available
