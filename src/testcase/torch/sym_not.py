import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.sym_not)
class TorchSymUnotTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_sym_not_correctness(self):
        # Testing torch.sym_not with a SymBool
        random_bool_val = random.choice([True, False])  # Generate a random boolean value
        result = torch.sym_not(random_bool_val)
        return result
