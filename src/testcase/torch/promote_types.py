import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.promote_types)
class TorchPromoteUtypesTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_promote_types_correctness(self):
        # Randomly select two dtypes from the list of supported dtypes
        dtypes = [torch.float32, torch.float64, torch.int32, torch.int64, torch.uint8, torch.int16, torch.bool]
        type1 = random.choice(dtypes)
        type2 = random.choice(dtypes)
        result = torch.promote_types(type1, type2)
        return result
