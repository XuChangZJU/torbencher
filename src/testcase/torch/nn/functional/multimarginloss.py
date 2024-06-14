import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.multimarginloss)
class TorchNnFunctionalMultimarginlossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_multi_margin_loss_correctness(self):
    # Randomly generate dimensions for input tensor
    batch_size = random.randint(1, 5)  # Random batch size
    num_classes = random.randint(2, 10)  # Random number of classes

    # Generate random input tensor with shape (batch_size, num_classes)
    input_tensor = torch.randn(batch_size, num_classes)

    # Generate random target tensor with shape (batch_size,)
    target_tensor = torch.randint(0, num_classes, (batch_size,))

    # Randomly choose p value (1 or 2)
    p = random.choice([1, 2])

    # Randomly choose margin value between 0.1 and 10.0
    margin = random.uniform(0.1, 10.0)

    # Compute multi_margin_loss
    result = torch.nn.functional.multi_margin_loss(input_tensor, target_tensor, p, margin)
    return result
