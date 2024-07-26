import torch
import random
from torch.utils.tensorboard import SummaryWriter

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.utils.tensorboard.writer.flush)
class TorchUtilsTensorboardWriterFlushTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_flush_correctness(self):
        # Create a SummaryWriter instance
        log_dir = f"runs/test_{random.randint(1, 1000)}"
        writer = SummaryWriter(log_dir=log_dir)

        # Add some random scalar data
        for i in range(random.randint(1, 10)):
            writer.add_scalar('test_scalar', random.uniform(0.1, 10.0), i)

        # Flush the writer to ensure all pending events have been written to disk
        writer.flush()

        # Close the writer to release resources
        writer.close()

        # Return the log directory to verify the flush effect
        return log_dir
