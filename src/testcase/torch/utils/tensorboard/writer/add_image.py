import torch
import random
from torch.utils.tensorboard import SummaryWriter

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.utils.tensorboard.writer.add_image)
class TorchUtilsTensorboardWriterAddimageTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_add_image_correctness(self):
        writer = SummaryWriter()

        # Randomly generate image dimensions
        channels = random.randint(1, 3)  # Random number of channels (1 for grayscale, 3 for RGB)
        height = random.randint(32, 256)  # Random height between 32 and 256 pixels
        width = random.randint(32, 256)  # Random width between 32 and 256 pixels

        # Create a random image tensor
        image_tensor = torch.randn(channels, height, width)

        # Randomly generate step
        step = random.randint(0, 100)

        # Add image to tensorboard
        writer.add_image('random_image', image_tensor, step)

        # Close the writer
        writer.close()
