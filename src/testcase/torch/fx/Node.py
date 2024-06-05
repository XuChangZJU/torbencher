
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.fx.Node)
class TorchFxNodeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.11")
    def test_node_correctness(self):
        # TODO: No concrete test case for Node.type due to its abstract nature.
        return None

    @test_api_version.larger_than("1.11")
    def test_node_large_scale(self):
        # TODO: No concrete test case for Node.type due to its abstract nature.
        return None


