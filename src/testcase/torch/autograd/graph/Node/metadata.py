import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.autograd.graph.Node.metadata)
class TorchAutogradGraphNodeMetadataTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_metadata_correctness(self):
        # No information on how to construct a Node object is available, 
        # so this test case cannot be generated.
        return None
