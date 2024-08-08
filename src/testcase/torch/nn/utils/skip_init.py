import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.utils.skip_init)
class TorchNnUtilsSkipUinitTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_skip_init_correctness(self):
        # Randomly choose the input and output features for the Linear layer
        in_features = random.randint(1, 10)
        out_features = random.randint(1, 10)

        # Instantiate the Linear layer using skip_init
        linear_layer = torch.nn.utils.skip_init(torch.nn.Linear, in_features, out_features)
