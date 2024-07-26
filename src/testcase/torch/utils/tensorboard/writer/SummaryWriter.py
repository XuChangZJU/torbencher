import torch
import random
from torch.utils.tensorboard import SummaryWriter

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.utils.tensorboard.writer.SummaryWriter)
class TorchUtilsTensorboardWriterSummarywriterTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_summary_writer_scalar(self):
        log_dir = f"runs/test_{random.randint(1, 1000)}"  # Random log directory
        writer = SummaryWriter(log_dir)

        tag = f"test_scalar_{random.randint(1, 1000)}"  # Random tag
        scalar_value = random.uniform(0.1, 100.0)  # Random scalar value between 0.1 and 100.0
        global_step = random.randint(1, 100)  # Random global step

        writer.add_scalar(tag, scalar_value, global_step)
        writer.close()

        return log_dir
