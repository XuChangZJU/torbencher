import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.functional.feature_alpha_dropout)
class TorchNnFunctionalFeaturealphadropoutTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_feature_alpha_dropout_correctness(self):
        """
        Test the correctness of torch.nn.functional.feature_alpha_dropout with small scale random parameters.
        """
        dim = random.randint(2, 4)  # Random dimension for the tensors, at least 2 dimensions are required.
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        input_tensor = torch.randn(input_size)  # Random input tensor
        p = random.uniform(0.1, 0.9)  # Random p value between 0.1 and 0.9
        training = random.choice([True, False])  # Randomly choose training mode

        result = torch.nn.functional.feature_alpha_dropout(input_tensor, p, training)
        return result
