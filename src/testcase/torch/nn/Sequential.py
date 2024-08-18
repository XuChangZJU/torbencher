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

        input_size = random.randint(4, 16)
        hidden_size = random.randint(4, 16)
        output_size = random.randint(4, 16)

        batch_size = random.randint(4, 16)
        input_tensor = torch.randn(batch_size, input_size)

        l1= torch.nn.Linear(input_size, hidden_size)
        with torch.no_grad():
            l1.weight.copy_(torch.randn(hidden_size, input_size) * 0.01)
            l1.bias.copy_(torch.randn(hidden_size) * 0.01)

        l2 = torch.nn.Linear(hidden_size, output_size)
        with torch.no_grad():
            l2.weight.copy_(torch.randn(output_size, hidden_size) * 0.01)
            l2.bias.copy_(torch.randn(output_size) * 0.01)

        model = torch.nn.Sequential(
            l1,
            torch.nn.ReLU(),
            l2
        )

        # Pass the input tensor through the Sequential model
        output_tensor = model(input_tensor)
        return output_tensor
