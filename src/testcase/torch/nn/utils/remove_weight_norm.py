import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.utils.remove_weight_norm)
class TorchNnUtilsRemoveweightnormTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_remove_weight_norm_correctness(self):
        # Random dimensions for the Linear layer
        in_features = random.randint(1, 10)
        out_features = random.randint(1, 10)

        # Create a Linear layer and apply weight normalization
        linear_layer = torch.nn.Linear(in_features, out_features)
        weight_normed_linear_layer = torch.nn.utils.weight_norm(linear_layer)

        # Remove weight normalization
        result = torch.nn.utils.remove_weight_norm(weight_normed_linear_layer)

        return result
