
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.CosineEmbeddingLoss)
class TorchNNCosineEmbeddingLossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cosine_embedding_loss(self):
        
        a = torch.randn(10, 5)
        b = torch.randn(10, 5)
        target = torch.randint(low=-1, high=2, size=(10,))
        loss = torch.nn.CosineEmbeddingLoss()
        result = loss(a, b, target)
        return result

