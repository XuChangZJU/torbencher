
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linspace)
class TorchLinspaceTestCase(```python
class TorchLinspaceTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_linspace(self):
        
        result = torch.linspace(3, 10, steps=5)
        return result

