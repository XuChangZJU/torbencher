import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.compile)
class TorchCompileTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_compile_correctness(self):
        @torch.compile(options={"triton.cudagraphs": True}, fullgraph=True)
        def foo(x):
            return torch.sin(x) + torch.cos(x)
