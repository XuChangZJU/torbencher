
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.is_bf16_supported)
class TorchCudaIsBf16SupportedTestCase(TorBencherTestCaseBase):
    def test_is_bf16_supported_0(self, input=None):
        if input is not None:
            result = torch.cuda.is_bf16_supported()
            return [result, input]
        result = torch.cuda.is_bf16_supported()
        return [result, None]

