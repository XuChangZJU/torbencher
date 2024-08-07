import torch
import random
from torch.utils.tensorboard import SummaryWriter

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.utils.tensorboard.writer.add_pr_curve)
class TorchUtilsTensorboardWriterAddUprUcurveTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_add_pr_curve_correctness(self):
        writer = SummaryWriter()

        num_thresholds = random.randint(1, 10)  # Random number of thresholds
        num_points = random.randint(1, 100)  # Random number of data points

        labels = torch.randint(0, 2, (num_points,))  # Random binary labels
        predictions = torch.rand(num_points)  # Random prediction scores between 0 and 1

        writer.add_pr_curve('pr_curve', labels, predictions, num_thresholds)
        writer.close()
