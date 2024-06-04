
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.hinge_embedding_loss)
class TorchNNFunctionalHingeEmbeddingLossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_hinge_embedding_loss_common(self):
        a = torch.randn(3, 2)
        b = torch.tensor([[-1, 1, 1], [-1, -1, 1]])
        c = 1.0
        d = True
        e = True
        f = 'mean'
        result = torch.nn.functional.hinge_embedding_loss(a, b, margin=c, size_average=d, reduce=e, reduction=f)
        return result


