import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.autograd.profiler_util.StringTable)
class TorchAutogradProfilerUutilStringtableTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_string_table_correctness(self):
        # Since torch.autograd.profiler_util.StringTable does not take any parameters,
        # we will simply create an instance of it and check its type.
        string_table = torch.autograd.profiler_util.StringTable()
        return isinstance(string_table, torch.autograd.profiler_util.StringTable)
