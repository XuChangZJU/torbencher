
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.distributed.is_mpi_available)
class TorchDistributedIsMpiAvailableTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_is_mpi_available_0(self):
        
        result = torch.distributed.is_mpi_available()
        return result


