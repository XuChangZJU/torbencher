import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.utils.parametrize.register_parametrization)
class TorchNnUtilsParametrizeRegisterUparametrizationTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_register_parametrization_correctness(self):
        # Randomly generate dimensions for the linear layer
        in_features = random.randint(1, 10)
        out_features = random.randint(1, 10)

        # Define a simple parametrization that makes the weight matrix symmetric
        class Symmetric(nn.Module):
            def forward(self, X):
                return X.triu() + X.triu(1).T  # Return a symmetric matrix

            def right_inverse(self, A):
                return A.triu()

        # Create a linear layer
        linear_layer = nn.Linear(in_features, out_features)

        # Register the symmetric parametrization
        P.register_parametrization(linear_layer, "weight", Symmetric())

        # Check if the weight matrix is symmetric
        is_symmetric = torch.allclose(linear_layer.weight, linear_layer.weight.T)

        return is_symmetric
