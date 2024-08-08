import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.backends.mha.get_fastpath_enabled)
class TorchBackendsMhaGetUfastpathUenabledTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_get_fastpath_enabled(self):
        # Get the current state of fastpath
        fastpath_enabled = torch.backends.cuda.matmul.allow_tf32

        # Check if the returned value is a boolean
        assert isinstance(fastpath_enabled, bool), "The returned value should be a boolean"

        return fastpath_enabled
