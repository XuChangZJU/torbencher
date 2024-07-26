import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.triplet_margin_loss)
class TorchNnFunctionalTripletmarginlossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_triplet_margin_loss_correctness(self):
        # Randomly generate input tensors
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        anchor = torch.randn(input_size)
        positive = torch.randn(input_size)
        negative = torch.randn(input_size)
        # triplet_margin_loss is valid when margin > 0
        margin = random.uniform(0.1, 10.0)
        result = torch.nn.functional.triplet_margin_loss(anchor, positive, negative, margin)
        return result
    
    
    
    