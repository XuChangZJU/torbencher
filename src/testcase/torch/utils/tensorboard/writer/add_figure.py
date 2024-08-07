import torch
import random
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.utils.tensorboard.writer.add_figure)
class TorchUtilsTensorboardWriterAddUfigureTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_add_figure_correctness(self):
        # Create a random figure
        fig, ax = plt.subplots()
        x = torch.randn(100).numpy()
        y = torch.randn(100).numpy()
        ax.scatter(x, y)

        # Create a SummaryWriter
        writer = SummaryWriter()

        # Add the figure to the writer
        writer.add_figure('random_scatter', fig)

        # Close the writer
        writer.close()
