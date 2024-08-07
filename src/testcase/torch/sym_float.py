import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.sym_float)
class TorchSymUfloatTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_sym_float_correctness(self):
        # Create a random SymInt
        sym_int = torch.sym_int(random.randint(-100, 100))

        # Cast the SymInt to a SymFloat
        result = torch.sym_float(sym_int)
        return result
