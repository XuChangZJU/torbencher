import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.export.buffers)
class TorchExportBuffersTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_export_buffers_correctness(self):
        # Randomly generate the number of buffers
        num_buffers = random.randint(1, 5)
        
        # Create a list of random tensors to act as buffers
        buffers = [torch.randn(random.randint(1, 4), random.randint(1, 5)) for _ in range(num_buffers)]
        
        # Create a dummy module to hold the buffers
        class DummyModule(torch.nn.Module):
            def __init__(self, buffers):
                super(DummyModule, self).__init__()
                for i, buffer in enumerate(buffers):
                    self.register_buffer(f'buffer_{i}', buffer)
        
        dummy_module = DummyModule(buffers)
        
        # Export the buffers
        exported_buffers = {name: buffer for name, buffer in dummy_module.named_buffers()}
        
        return exported_buffers
    