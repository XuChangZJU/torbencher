import random

import torch
import torch.nn as nn

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.utils.parametrizations.orthogonal)
class TorchNnUtilsParametrizationsOrthogonalTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_orthogonal_correctness(self):
        # Randomly generate dimensions for the weight matrix
        in_features = random.randint(1, 10)
        out_features = random.randint(1, 10)

        # Create a random linear layer
        linear_layer = nn.Linear(in_features, out_features)

        # Apply orthogonal parametrization
        orthogonal_layer = torch.nn.utils.parametrizations.orthogonal(linear_layer)

        # Retrieve the orthogonal weight matrix
        orthogonal_weight = orthogonal_layer.weight

        # Check orthogonality condition
        if in_features >= out_features:
            identity_matrix = torch.eye(out_features)
            result = torch.dist(orthogonal_weight.T @ orthogonal_weight, identity_matrix)
        else:
            identity_matrix = torch.eye(in_features)
            result = torch.dist(orthogonal_weight @ orthogonal_weight.T, identity_matrix)

        return result
