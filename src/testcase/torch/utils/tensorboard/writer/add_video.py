import torch
import random
from torch.utils.tensorboard import SummaryWriter

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.utils.tensorboard.writer.add_video)
class TorchUtilsTensorboardWriterAddvideoTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_add_video_correctness(self):
        writer = SummaryWriter()

        batch_size = random.randint(1, 4)  # Random batch size between 1 and 4
        channels = 3  # RGB channels
        frames = random.randint(1, 10)  # Random number of frames between 1 and 10
        height = random.randint(64, 128)  # Random height between 64 and 128
        width = random.randint(64, 128)  # Random width between 64 and 128

        video_tensor = torch.randn(batch_size, channels, frames, height, width)  # Random video tensor
        global_step = random.randint(0, 100)  # Random global step between 0 and 100

        writer.add_video('random_video', video_tensor, global_step)
        writer.close()
