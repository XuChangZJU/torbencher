
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.EmbeddingBag)
class TorchEmbeddingBagTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_embeddingbag_correctness(self):
        num_embeddings = random.randint(1, 10)
        embedding_dim = random.randint(1, 10)
        input_tensor = torch.randint(0, num_embeddings, (random.randint(1, 10),))
        offsets = torch.randint(1, 10, (random.randint(1, 10),))
        embedding_bag = torch.nn.EmbeddingBag(num_embeddings, embedding_dim)
        result = embedding_bag(input_tensor, offsets)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_embeddingbag_large_scale(self):
        num_embeddings = random.randint(1000, 10000)
        embedding_dim = random.randint(100, 1000)
        input_tensor = torch.randint(0, num_embeddings, (random.randint(1000, 10000),))
        offsets = torch.randint(100, 1000, (random.randint(1000, 10000),))
        embedding_bag = torch.nn.EmbeddingBag(num_embeddings, embedding_dim)
        result = embedding_bag(input_tensor, offsets)
        return result

