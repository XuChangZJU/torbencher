
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.EmbeddingBag)
class TorchNNEmbeddingBagTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_embedding_bag(self):
        a = torch.randint(0, 10, (2, 3))
        embedding_bag = torch.nn.EmbeddingBag(10, 3)
        result = embedding_bag(a)
        return result

