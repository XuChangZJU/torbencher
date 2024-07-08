import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.autograd.profiler_util.StringTable)
class TorchAutogradProfilerutilStringtableTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_string_table_correctness(self):
        # Since torch.autograd.profiler_util.StringTable does not take any parameters,
        # we will simply create an instance of it and check its type.
        string_table = torch.autograd.profiler_util.StringTable()
        return isinstance(string_table, torch.autograd.profiler_util.StringTable)
    