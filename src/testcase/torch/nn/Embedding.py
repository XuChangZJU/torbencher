import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.Embedding)
class TorchNnEmbeddingTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_embedding_correctness(self):
        # Randomly generate the size of the dictionary of embeddings
        num_embeddings = random.randint(5, 20)
        # Randomly generate the size of each embedding vector
        embedding_dim = random.randint(3, 10)
        
        # Create the Embedding layer
        embedding_layer = torch.nn.Embedding(num_embeddings, embedding_dim)
        
        # Randomly generate the input indices
        batch_size = random.randint(1, 5)
        sequence_length = random.randint(1, 10)
        input_indices = torch.randint(0, num_embeddings, (batch_size, sequence_length), dtype=torch.long)
        
        # Get the embeddings for the input indices
        result = embedding_layer(input_indices)
        return result
    