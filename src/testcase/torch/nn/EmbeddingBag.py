import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.EmbeddingBag)
class TorchNnEmbeddingbagTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_embedding_bag_correctness(self):
    num_embeddings = random.randint(5, 20)  # Random number of embeddings
    embedding_dim = random.randint(2, 10)  # Random embedding dimension
    mode = random.choice(['sum', 'mean', 'max'])  # Random mode selection

    embedding_bag = torch.nn.EmbeddingBag(num_embeddings, embedding_dim, mode=mode)

    # Generate random input tensor with indices
    input_length = random.randint(5, 15)  # Random length of input tensor
    input_indices = torch.randint(0, num_embeddings, (input_length,), dtype=torch.long)

    # Generate random offsets tensor
    num_offsets = random.randint(1, input_length // 2)  # Random number of offsets
    offsets = torch.randint(0, input_length, (num_offsets,), dtype=torch.long)
    offsets = torch.cat((offsets, torch.tensor([input_length], dtype=torch.long)))  # Ensure last offset is input_length

    result = embedding_bag(input_indices, offsets)
    return result
