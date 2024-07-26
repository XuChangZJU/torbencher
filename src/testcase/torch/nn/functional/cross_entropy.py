import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.functional.cross_entropy)
class TorchNnFunctionalCrossentropyTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cross_entropy_correctness(self):
        # Randomly generate input shape
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random input tensor
        input_tensor = torch.randn(input_size)

        # Generate random target tensor with class indices
        num_classes = random.randint(1, input_size[
            -1])  # Number of classes should be less than or equal to the last dimension of input
        target_tensor_indices = torch.randint(0, num_classes, input_size[:-1], dtype=torch.int64)

        # Calculate cross entropy loss with class indices
        result_indices = torch.nn.functional.cross_entropy(input_tensor, target_tensor_indices)

        # Generate random target tensor with class probabilities
        target_tensor_probs = torch.randn(input_size).softmax(dim=-1)  # Apply softmax to ensure probabilities sum to 1

        # Calculate cross entropy loss with class probabilities
        result_probs = torch.nn.functional.cross_entropy(input_tensor, target_tensor_probs)

        return result_indices, result_probs
