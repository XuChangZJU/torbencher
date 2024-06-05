
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.einsum)
class TorchEinsumTestCase(TorBencherTestCaseBase):

import random
import torch

