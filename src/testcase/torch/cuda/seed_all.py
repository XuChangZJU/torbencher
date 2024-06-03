
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.seed_all)
class TorchCudaSeedAllTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_seed_all(self, input=None):
        if input is not None:
            result = torch.cuda.seed_all()
            return result
        result = torch.cuda.seed_all()
        return result


