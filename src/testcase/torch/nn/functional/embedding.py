
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.embedding)
class TorchNNFunctionalEmbeddingTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_embedding_common(self):
        
        a = torch.tensor([[1, 2, 4, 5, 4, 3, 2, 9]])
        b = torch.randn(10, 3)
        c = None
        d = None
        e = 2.0
        f = False
        g = False
        result = torch.nn.functional.embedding(a, b, padding_idx=c, max_norm=d, norm_type=e, scale_grad_by_freq=f, sparse=g)
        return result


