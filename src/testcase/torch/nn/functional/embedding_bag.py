
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.embedding_bag)
class TorchNNFunctionalEmbeddingBagTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_embedding_bag_common(self):
        
        a = torch.tensor([1, 2, 4, 2, 0, 3])
        b = torch.randn(5, 3)
        c = torch.tensor([0, 2])
        d = None
        e = 2.0
        f = False
        g = 'sum'
        h = False
        i = None
        result = torch.nn.functional.embedding_bag(a, b, c, max_norm=d, norm_type=e, scale_grad_by_freq=f, mode=g, sparse=h, per_sample_weights=i)
        return result


