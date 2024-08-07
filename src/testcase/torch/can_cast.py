import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.can_cast)
class TorchCanUcastTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_can_cast_correctness(self):
        # Define the list of all possible torch data types
        dtypes = [
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.float16,
            torch.float32, torch.float64, torch.complex64, torch.complex128, torch.bool
        ]

        # Randomly select source and target dtypes from the list
        src_dtype = random.choice(dtypes)
        tgt_dtype = random.choice(dtypes)

        # Test if torch.can_cast correctly determines the casting possibility
        result = torch.can_cast(src_dtype, tgt_dtype)
        return result
