import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.utils.rename_privateuse1_backend)
class TorchUtilsRenameprivateuse1backendTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_rename_privateuse1_backend_correctness(self):
        backend_name = "foo"  # backend_name: str
        torch.utils.rename_privateuse1_backend(backend_name)
        return torch.utils.rename_privateuse1_backend
    
    
    
    