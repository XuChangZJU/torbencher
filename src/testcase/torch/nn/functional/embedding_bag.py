import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.functional.embedding_bag)
class TorchNnFunctionalEmbeddingUbagTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_embedding_bag_correctness(self):
        # Randomly generate the number of embeddings and embedding dimension
        num_embeddings = random.randint(5, 20)
        embedding_dim = random.randint(3, 10)

        # Randomly generate the embedding matrix
        embedding_matrix = torch.randn(num_embeddings, embedding_dim)

        # Randomly generate the input tensor with indices into the embedding matrix
        input_length = random.randint(5, 15)
        input_tensor = torch.randint(0, num_embeddings, (input_length,), dtype=torch.long)

        # Randomly generate the offsets tensor
        num_bags = random.randint(1, input_length)
        offsets = torch.randint(0, input_length, (num_bags,), dtype=torch.long)
        offsets = torch.sort(offsets)[0]  # Ensure offsets are sorted

        # Ensure the first offset is 0
        offsets[0] = 0

        # Compute the embedding bag
        result = torch.nn.functional.embedding_bag(input_tensor, embedding_matrix, offsets)
        return result
