import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.functional.multilabel_soft_margin_loss)
class TorchNnFunctionalMultilabelUsoftUmarginUlossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_multilabel_soft_margin_loss_correctness(self):
        # Define the dimensions for the input tensors
        dim1 = random.randint(1, 10)
        dim2 = random.randint(1, 10)
        # Generate random input tensor 
        input = torch.randn(dim1, dim2)
        # Generate random target tensor with values -1, 0, 1
        target = torch.randint(-1, 2, (dim1, dim2), dtype=torch.float)
        result = torch.nn.functional.multilabel_soft_margin_loss(input, target)
        return result
