import random

import torch
import torch.nn as nn

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.utils.parametrizations.orthogonal)
class TorchNnUtilsParametrizationsOrthogonalTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_orthogonal_correctness(self):
        orth_linear = torch.nn.utils.parametrizations.orthogonal(nn.Linear(20, 40))
        Q = orth_linear.weight
        return torch.dist(Q.T @ Q, torch.eye(20))