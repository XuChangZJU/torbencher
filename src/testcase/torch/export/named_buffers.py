import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.export.named_buffers)
class TorchExportNamedbuffersTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_named_buffers_correctness(self):
        class RandomModel(torch.nn.Module):
            def __init__(self):
                super(RandomModel, self).
                self.buffer1 = torch.randn(random.randint(1, 4), random.randint(1, 5))
                self.buffer2 = torch.randn(random.randint(1, 4), random.randint(1, 5))
            
            def forward(self, x):
                return x
    
        model = RandomModel()
        result = list(model.named_buffers())
        return result
    