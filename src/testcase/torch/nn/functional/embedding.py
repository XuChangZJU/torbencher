import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.functional.embedding)
class TorchNnFunctionalEmbeddingTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_embedding_correctness(self):
        # Randomly generate input shape
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random input tensor
        input = torch.randint(0, 10, input_size)  # Input tensor containing indices, upper bound is exclusive

        # Generate random embedding matrix
        embedding_matrix = torch.randn(10,
                                       3)  # embedding matrix of size V x embedding_dim, here V = 10, embedding_dim = 3

        result = torch.nn.functional.embedding(input, embedding_matrix)
        return result
