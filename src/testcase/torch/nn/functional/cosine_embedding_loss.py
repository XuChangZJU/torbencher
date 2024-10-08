import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.functional.cosine_embedding_loss)
class TorchNnFunctionalCosineUembeddingUlossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_cosine_embedding_loss_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        # Generate random tensors for input1 and input2
        input1 = torch.randn(input_size)
        input2 = torch.randn(input_size)

        # Generate random target tensor with values -1 or 1
        target = torch.randint(0, 2, (num_of_elements_each_dim,)).float() * 2 - 1

        # Calculate cosine embedding loss
        result = torch.nn.functional.cosine_embedding_loss(input1.view(num_of_elements_each_dim, -1),
                                                           input2.view(num_of_elements_each_dim, -1),
                                                           target)
        return result
