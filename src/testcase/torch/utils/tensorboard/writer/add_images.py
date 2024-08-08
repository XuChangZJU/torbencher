import random

import torch
from torch.utils.tensorboard import SummaryWriter

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.utils.tensorboard.writer.add_images)
class TorchUtilsTensorboardWriterAddUimagesTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_add_images_correctness(self):
        writer = SummaryWriter()

        # Randomly generate the number of images
        num_images = random.randint(1, 10)

        # Randomly generate the number of channels (1 for grayscale, 3 for RGB)
        num_channels = random.choice([1, 3])

        # Randomly generate the height and width of the images
        height = random.randint(32, 256)
        width = random.randint(32, 256)

        # Create a random tensor with the shape (num_images, num_channels, height, width)
        images = torch.randn(num_images, num_channels, height, width)

        # Randomly generate a global step
        global_step = random.randint(0, 100)

        # Add images to the writer
        writer.add_images('test_images', images, global_step)

        # Close the writer
        writer.close()
