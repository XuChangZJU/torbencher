import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.CosineEmbeddingLoss)
class TorchNnCosineembeddinglossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cosine_embedding_loss_correctness(self):
        # Random batch size between 1 and 4
        batch_size = random.randint(1, 4)
        # Random embedding dimension between 1 and 5
        embedding_dim = random.randint(1, 5)

        # Generate random input tensors with the same shape
        input1 = torch.randn(batch_size, embedding_dim, requires_grad=True)
        input2 = torch.randn(batch_size, embedding_dim, requires_grad=True)

        # Generate random target tensor with values 1 or -1
        target = torch.randint(0, 2, (batch_size,)).float() * 2 - 1

        # Create the CosineEmbeddingLoss criterion
        criterion = torch.nn.CosineEmbeddingLoss()

        # Compute the loss
        loss = criterion(input1, input2, target)

        return loss
