import torch
import random
from torch.utils.tensorboard import SummaryWriter

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.utils.tensorboard.writer.add_audio)
class TorchUtilsTensorboardWriterAddUaudioTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_add_audio_correctness(self):
        writer = SummaryWriter()

        # Randomly generate sample rate between 8000 and 48000
        sample_rate = random.randint(8000, 48000)

        # Randomly generate number of channels between 1 and 2
        num_channels = random.randint(1, 2)

        # Randomly generate number of samples between 1 and 10000
        num_samples = random.randint(1, 10000)

        # Generate random audio tensor with shape (num_channels, num_samples)
        audio_tensor = torch.randn(num_channels, num_samples)

        # Randomly generate step
        step = random.randint(0, 100)

        # Add audio to tensorboard
        writer.add_audio('test_audio', audio_tensor, step, sample_rate)

        # Close the writer
        writer.close()
