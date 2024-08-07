import torch
import random
import torch.utils.checkpoint

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.utils.checkpoint.set_checkpoint_debug_enabled)
class TorchUtilsCheckpointSetUcheckpointUdebugUenabledTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_set_checkpoint_debug_enabled_correctness(self):
        # Randomly enable or disable checkpoint debug mode
        enable_debug = random.choice([True, False])

        # Set checkpoint debug mode
        previous_state = torch.utils.checkpoint.set_checkpoint_debug_enabled(enable_debug)

        return previous_state
