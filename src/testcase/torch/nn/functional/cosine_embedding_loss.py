
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.cosine_embedding_loss)
class TorchNNFunctionalCosineEmbeddingLossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cosine_embedding_loss_common(self):
        
        a = torch.randn(100, 128)
        b = torch.randn(100, 128)
        c = torch.ones(100)
        d = 0.5
        e = True
        f = True
        g = 'mean'
        result = torch.nn.functional.cosine_embedding_loss(a, b, c, margin=d, size_average=e, reduce=f, reduction=g)
        return result


