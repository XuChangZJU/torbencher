
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.CosineEmbeddingLoss)
class TorchCosineEmbeddingLossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cosineembeddingloss_correctness(self):
        input1 = torch.randn(random.randint(1, 10), random.randint(1, 10))
        input2 = torch.randn(random.randint(1, 10), random.randint(1, 10))
        target = torch.randint(0, 2, (random.randint(1, 10),), dtype=torch.long)
        cosine_embedding_loss = torch.nn.CosineEmbeddingLoss()
        result = cosine_embedding_loss(input1, input2, target)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_cosineembeddingloss_large_scale(self):
        input1 = torch.randn(random.randint(1000, 10000), random.randint(100, 1000))
        input2 = torch.randn(random.randint(1000, 10000), random.randint(100, 1000))
        target = torch.randint(0, 2, (random.randint(1000, 10000),), dtype=torch.long)
        cosine_embedding_loss = torch.nn.CosineEmbeddingLoss()
        result = cosine_embedding_loss(input1, input2, target)
        return result

