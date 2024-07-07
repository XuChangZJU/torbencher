import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.backends.mkldnn.is_available)
class TorchBackendsMkldnnIsavailableTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_is_available_correctness(self):
        # No specific input parameters needed for is_available()
        result_mkldnn_available = torch.backends.mkldnn.is_available()
        return result_mkldnn_available
    
    
    
    