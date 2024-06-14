import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.CrossEntropyLoss)
class TorchNnCrossentropylossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cross_entropy_loss_correctness(self):
        # Randomly generate the number of classes
        num_classes = random.randint(2, 10)
        
        # Randomly generate the batch size
        batch_size = random.randint(1, 5)
        
        # Randomly generate the input tensor size
        input_size = [batch_size, num_classes]
        
        # Generate random input tensor with unnormalized logits
        input_tensor = torch.randn(input_size, requires_grad=True)
        
        # Generate random target tensor with class indices in the range [0, num_classes)
        target_tensor = torch.empty(batch_size, dtype=torch.long).random_(num_classes)
        
        # Initialize CrossEntropyLoss
        loss_fn = torch.nn.CrossEntropyLoss()
        
        # Compute the loss
        loss = loss_fn(input_tensor, target_tensor)
        
        return loss
    