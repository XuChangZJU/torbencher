
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.Embedding)
class TorchEmbeddingTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_embedding_correctness(self):
        num_embeddings = random.randint(1, 10)
        embedding_dim = random.randint(1, 10)
        input_tensor = torch.randint(0, num_embeddings, (random.randint(1, 10),))
        embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
        result = embedding(input_tensor)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_embedding_large_scale(self):
        num_embeddings = random.randint(1000, 10000)
        embedding_dim = random.randint(100, 1000)
        input_tensor = torch.randint(0, num_embeddings, (random.randint(1000, 10000),))
        embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
        result = embedding(input_tensor)
        return result

