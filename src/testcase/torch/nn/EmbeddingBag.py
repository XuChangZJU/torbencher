import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api
import unittest

@test_api(torch.nn.EmbeddingBag)
class TorchNnEmbeddingbagTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_embedding_bag_correctness(self):
        num_embeddings = random.randint(5, 20)  # Random number of embeddings
        embedding_dim = random.randint(2, 10)  # Random embedding dimension
        mode = random.choice(['sum', 'mean', 'max'])  # Random mode selection

        embedding_bag = torch.nn.EmbeddingBag(num_embeddings, embedding_dim, mode=mode)

        with torch.no_grad():
            embedding_bag.weight = torch.nn.Parameter(torch.randn(num_embeddings, embedding_dim))

        # Generate random input tensor with indices
        input_length = random.randint(5, 15)  # Random length of input tensor
        input_indices = torch.randint(0, num_embeddings, (input_length,), dtype=torch.long)

        # Define num_offsets and generate correct offsets tensor starting from 0 and ending with input_length
        num_offsets = random.randint(1,
                                     input_length)  # Ensure we have at least one and not more offsets than input_length
        offsets = torch.arange(0, input_length + 1,
                               step=(input_length // num_offsets) if input_length > num_offsets else 1,
                               dtype=torch.long)
        # Ensure the last offset is exactly input_length
        if offsets[-1] != input_length:
            offsets[-1] = input_length

        result = embedding_bag(input_indices, offsets)
        return result
