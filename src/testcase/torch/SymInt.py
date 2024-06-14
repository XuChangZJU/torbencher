import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.symint)
class TorchSymintTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_sym_int_correctness(self):
    # Create a random SymInt
    sym_int = torch.sym_int(random.randint(-1000, 1000))  # Generate a random integer between -1000 and 1000
    result = torch.sym_int(sym_int)
    return result
