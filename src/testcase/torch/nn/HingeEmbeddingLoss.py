import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.HingeEmbeddingLoss)
class TorchNnHingeembeddinglossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_hinge_embedding_loss_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]
    
        # Random input tensor
        input_tensor = torch.randn(input_size)
        # Random labels tensor containing 1 or -1
        labels_tensor = torch.randint(0, 2, input_size) * 2 - 1
        # Random margin value between 0.1 and 10.0
        margin = random.uniform(0.1, 10.0)
    
        # Initialize HingeEmbeddingLoss with random margin
        loss_fn = torch.nn.HingeEmbeddingLoss(margin=margin)
    
        # Compute the loss
        loss = loss_fn(input_tensor, labels_tensor)
        return loss
    
    
    
    