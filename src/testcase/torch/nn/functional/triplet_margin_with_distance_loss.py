import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.functional.triplet_margin_with_distance_loss)
class TorchNnFunctionalTripletUmarginUwithUdistanceUlossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_triplet_margin_with_distance_loss_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        # Generate random tensors for anchor, positive, and negative
        anchor = torch.randn(input_size)
        positive = torch.randn(input_size)
        negative = torch.randn(input_size)

        # Define a simple distance function (Euclidean distance)
        def distance_function(x, y):
            return torch.norm(x - y, p=2, dim=-1)

        # Compute the triplet margin loss
        result = torch.nn.functional.triplet_margin_with_distance_loss(anchor, positive, negative,
                                                                       distance_function=distance_function)
        return result
