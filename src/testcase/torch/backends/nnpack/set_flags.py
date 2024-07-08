import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.backends.nnpack.set_flags)
class TorchBackendsNnpackSetflagsTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_set_flags_correctness(self):
        # Randomly generate boolean values for the flags
        enabled = bool(random.randint(0, 1))
        auto = bool(random.randint(0, 1))
        
        # Set the flags using the generated boolean values
        torch.backends.quantized.engine = 'qnnpack' if enabled else 'none'
        
        # Return the current state of the flags to verify correctness
        return torch.backends.quantized.engine
    