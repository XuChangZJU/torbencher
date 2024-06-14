import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.MultiLabelMarginLoss)
class TorchNnMultilabelmarginlossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_multilabel_margin_loss_correctness(self):
        # Random batch size between 1 and 4
        batch_size = random.randint(1, 4)
        # Random number of classes between 2 and 5
        num_classes = random.randint(2, 5)
        
        # Generate random input tensor with shape (batch_size, num_classes)
        input_tensor = torch.randn(batch_size, num_classes)
        
        # Generate random target tensor with shape (batch_size, num_classes)
        # Ensure target values are valid indices and padded with -1
        target_tensor = torch.full((batch_size, num_classes), -1, dtype=torch.long)
        for i in range(batch_size):
            num_targets = random.randint(1, num_classes)
            target_indices = random.sample(range(num_classes), num_targets)
            target_tensor[i, :num_targets] = torch.tensor(target_indices, dtype=torch.long)
        
        # Initialize the MultiLabelMarginLoss criterion
        criterion = torch.nn.MultiLabelMarginLoss()
        
        # Compute the loss
        loss = criterion(input_tensor, target_tensor)
        
        return loss
    