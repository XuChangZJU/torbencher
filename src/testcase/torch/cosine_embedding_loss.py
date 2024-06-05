
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cosine_embedding_loss)
class TorchCosineEmbeddingLossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cosine_embedding_loss_correctness(self):
        dim = random.randint(1, 10)
        input1 = torch.randn(dim)
        input2 = torch.randn(dim)
        target = torch.randint(0, 2, (dim,))
        margin = random.uniform(0.1, 10.0)
        reduction = random.choice(['none', 'mean', 'sum'])
        result = torch.cosine_embedding_loss(input1, input2, target, margin=margin, reduction=reduction)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_cosine_embedding_loss_large_scale(self):
        dim = random.randint(1000, 10000)
        input1 = torch.randn(dim)
        input2 = torch.randn(dim)
        target = torch.randint(0, 2, (dim,))
        margin = random.uniform(0.1, 10.0)
        reduction = random.choice(['none', 'mean', 'sum'])
        result = torch.cosine_embedding_loss(input1, input2, target, margin=margin, reduction=reduction)
        return result

