import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api
import unittest


@test_api(torch.nn.Sequential)
class TorchNnSequentialTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_sequential_correctness(self):
        torch.manual_seed(0)
        random.seed(0)

        input_size = random.randint(4, 16)
        hidden_size = random.randint(4, 16)
        output_size = random.randint(4, 16)

        batch_size = random.randint(4, 16)
        input_tensor = torch.randn(batch_size, input_size)

        model = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size)
        )

        # Pass the input tensor through the Sequential model
        output_tensor = model(input_tensor)
        return output_tensor
