import random
from collections import namedtuple

import numpy as np
import torch
import torch.utils.data

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.utils.data.default_convert)
class TorchUtilsDataDefaultUconvertTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_default_convert_correctness(self):
        return torch.utils.data.default_convert([0, np.array([0, 1]), np.array([2, 3])])
