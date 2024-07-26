import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.backends.mha.get_fastpath_enabled)
class TorchBackendsMhaGetfastpathenabledTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_get_fastpath_enabled(self):
        # Get the current state of fastpath
        fastpath_enabled = torch.backends.cuda.matmul.allow_tf32

        # Check if the returned value is a boolean
        assert isinstance(fastpath_enabled, bool), "The returned value should be a boolean"

        return fastpath_enabled
